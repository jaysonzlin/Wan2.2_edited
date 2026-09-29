import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _write_full_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin"):
        (path / name).write_text("state\n")
    for rank in range(8):
        (path / f"random_states_{rank}.pkl").write_text("state\n")


def _write_four_gpu_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin"):
        (path / name).write_text("state\n")
    for rank in range(4):
        (path / f"random_states_{rank}.pkl").write_text("state\n")


def _prepare_launcher(tmp_path: Path) -> tuple[dict[str, str], Path]:
    (tmp_path / "configs/accelerate").mkdir(parents=True)
    (tmp_path / "configs/accelerate/h200_8gpu_2node.yaml").write_text("{}\n")
    (tmp_path / "configs/accelerate/h200_4gpu.yaml").write_text("{}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mamba_prefix = tmp_path / "mamba"
    (mamba_prefix / "bin").mkdir(parents=True)
    call_log = tmp_path / "calls.txt"

    _write_executable(bin_dir / "scontrol", "#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n")
    _write_executable(
        bin_dir / "getent", "#!/bin/bash\nprintf '10.0.0.1 STREAM node-a\\n'\n"
    )
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "ibstat", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "module",
        "#!/bin/bash\nprintf 'module %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )
    _write_executable(
        bin_dir / "srun",
        "#!/bin/bash\n"
        "printf 'srun %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "while (($#)); do\n"
        "  if [[ $1 == --export=* ]]; then\n"
        "    IFS=, read -ra exports <<< \"${1#--export=}\"\n"
        "    for entry in \"${exports[@]}\"; do\n"
        "      [[ $entry == *=* ]] && export \"$entry\"\n"
        "    done\n"
        "  fi\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(
        mamba_prefix / "bin/python",
        "#!/bin/bash\nprintf 'MAMBA_PREFLIGHT_OK cuda_devices=4\\n'\n",
    )
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\n"
        "printf 'distributed_debug=%s\\n' \"${TORCH_DISTRIBUTED_DEBUG:-unset}\" >> \"$CALL_LOG\"\n"
        "printf 'accelerate %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )

    return (
        os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_DIR": str(tmp_path),
            "MAMBA_ENV_PREFIX": str(mamba_prefix),
            "CALL_LOG": str(call_log),
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "node-a,node-b",
            "SLURM_NODEID": "0",
        },
        call_log,
    )


def test_frozen_video_launcher_uses_starting_checkpoint_for_an_empty_stage(
    tmp_path: Path,
) -> None:
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_pc200k_frozen_video.sh"
    assert script.is_file()
    starter = tmp_path / "source" / "checkpoint-12000"
    _write_full_checkpoint(starter)
    env, call_log = _prepare_launcher(tmp_path)

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=env | {"STARTING_CHECKPOINT": str(starter)},
    )

    assert result.returncode == 0, result.stderr
    assert "distributed_debug=INFO" in call_log.read_text()
    assert f"Resume setting: starter checkpoint {starter}" in result.stdout
    assert (
        "joint_simgen_frozen_video.py "
        "--config configs/train/joint_simgen_480_8gpu_pc200k_frozen_video.yaml "
        f"training.resume_from_checkpoint={starter}"
    ) in call_log.read_text()


def test_frozen_video_launcher_prefers_its_own_latest_checkpoint(
    tmp_path: Path,
) -> None:
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_pc200k_frozen_video.sh"
    assert script.is_file()
    starter = tmp_path / "source" / "checkpoint-12000"
    _write_full_checkpoint(starter)
    _write_full_checkpoint(
        tmp_path / "outputs/joint_simgen_8gpu_pc_bridge_frozen_video/checkpoint-13000"
    )
    env, call_log = _prepare_launcher(tmp_path)

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=env | {"STARTING_CHECKPOINT": str(starter)},
    )

    assert result.returncode == 0, result.stderr
    assert "Resume setting: latest frozen-video checkpoint" in result.stdout
    calls = call_log.read_text()
    assert "distributed_debug=INFO" in calls
    assert (
        "training.resume_from_checkpoint="
        f"{tmp_path}/outputs/joint_simgen_8gpu_pc_bridge_frozen_video/checkpoint-13000"
    ) in calls
    assert str(starter) not in calls


def test_single_node_frozen_video_launcher_uses_four_gpu_configuration(
    tmp_path: Path,
) -> None:
    script = PROJECT_DIR / "submit_joint_simgen_4gpu_pc200k_frozen_video.sh"
    starter = tmp_path / "source" / "checkpoint-12000"
    _write_full_checkpoint(starter)
    env, call_log = _prepare_launcher(tmp_path)

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=env | {"STARTING_CHECKPOINT": str(starter)},
    )

    assert result.returncode == 0, result.stderr
    assert f"Resume setting: starter checkpoint {starter}" in result.stdout
    assert (
        "accelerate launch --config_file configs/accelerate/h200_4gpu.yaml "
        "joint_simgen_frozen_video.py "
        "--config configs/train/joint_simgen_480_4gpu_pc200k_frozen_video.yaml "
        f"training.resume_from_checkpoint={starter}"
    ) in call_log.read_text()


def test_single_node_launcher_resumes_only_its_own_four_rank_checkpoint(
    tmp_path: Path,
) -> None:
    script = PROJECT_DIR / "submit_joint_simgen_4gpu_pc200k_frozen_video.sh"
    starter = tmp_path / "source" / "checkpoint-12000"
    _write_full_checkpoint(starter)
    _write_full_checkpoint(
        tmp_path / "outputs/joint_simgen_8gpu_pc_bridge_frozen_video/checkpoint-13000"
    )
    four_gpu_checkpoint = (
        tmp_path / "outputs/joint_simgen_4gpu_pc_bridge_frozen_video/checkpoint-14000"
    )
    _write_four_gpu_checkpoint(four_gpu_checkpoint)
    env, call_log = _prepare_launcher(tmp_path)

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=env | {"STARTING_CHECKPOINT": str(starter)},
    )

    assert result.returncode == 0, result.stderr
    assert "Resume setting: latest frozen-video checkpoint" in result.stdout
    assert f"training.resume_from_checkpoint={four_gpu_checkpoint}" in call_log.read_text()


def test_validation_launcher_passes_checkpoint_and_output_directory(tmp_path: Path) -> None:
    script = PROJECT_DIR / "submit_visualize_joint_simgen_frozen_video_validation.sh"
    checkpoint = tmp_path / "checkpoint-8000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_text("state\n")
    output_dir = tmp_path / "validation-videos"
    env, call_log = _prepare_launcher(tmp_path)
    _write_executable(
        tmp_path / "mamba/bin/python",
        "#!/bin/bash\nprintf 'python %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=env | {"CHECKPOINT_PATH": str(checkpoint), "OUTPUT_DIR": str(output_dir)},
    )

    assert result.returncode == 0, result.stderr
    assert f"Checkpoint: {checkpoint}" in result.stdout
    assert f"Output: {output_dir}" in result.stdout
    assert (
        "python visualize_joint_simgen_frozen_video_validation.py "
        "--config configs/train/joint_simgen_480_8gpu_pc200k_frozen_video.yaml "
        f"--checkpoint {checkpoint} --output-dir {output_dir}"
    ) in call_log.read_text()
