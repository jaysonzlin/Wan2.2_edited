import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_video_only_mamba_launcher_uses_isolated_eight_gpu_contract(tmp_path):
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_mamba_video_only_3slot.sh"
    assert script.is_file()

    (tmp_path / "configs/accelerate").mkdir(parents=True)
    (tmp_path / "configs/accelerate/h200_8gpu_2node.yaml").write_text("{}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mamba_prefix = tmp_path / "mamba"
    (mamba_prefix / "bin").mkdir(parents=True)
    call_log = tmp_path / "calls.txt"

    _write_executable(bin_dir / "scontrol", "#!/bin/bash\nprintf 'node-a\\n'\n")
    _write_executable(
        bin_dir / "getent", "#!/bin/bash\nprintf '10.0.0.1 STREAM node-a\\n'\n"
    )
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "ibstat", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "srun",
        "#!/bin/bash\nset -eu\n"
        "while (($#)); do\n"
        "  if [[ $1 == --export=* ]]; then\n"
        "    IFS=, read -ra exports <<< \"${1#--export=}\"\n"
        "    for entry in \"${exports[@]}\"; do [[ $entry == *=* ]] && export \"$entry\"; done\n"
        "  fi\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(mamba_prefix / "bin/python", "#!/bin/bash\nexit 0\n")
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_DIR": str(tmp_path),
            "MAMBA_ENV_PREFIX": str(mamba_prefix),
            "CALL_LOG": str(call_log),
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "node-a,node-b",
            "SLURM_NODEID": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Job ID: 12345" in result.stdout
    assert "Restart count: 0" in result.stdout
    assert "Nodes: node-a,node-b" in result.stdout
    assert "Rendezvous: 10.0.0.1:32345" in result.stdout
    assert f"Environment: {mamba_prefix}" in result.stdout
    assert (
        f"NCCL logs: {tmp_path}/logs/nccl-joint-simgen-video-only-3slot-12345"
        in result.stdout
    )
    assert "Resume setting: latest" in result.stdout
    assert "Start time:" in result.stdout
    assert "Node rank: 0; host:" in result.stdout
    calls = call_log.read_text()
    assert "--config_file configs/accelerate/h200_8gpu_2node.yaml" in calls
    assert "joint_simgen_video_only.py" in calls
    assert "--config configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml" in calls
    assert "data.train_start=0 data.train_end=127" in calls
    assert "data.validation_start=490 data.validation_end=499" in calls
    assert "validation.every_steps=1000 visualization.every_steps=1000" in calls
    assert "training.resume_from_checkpoint=latest" in calls
    assert "joint_simgen.py" not in calls


def test_h200_eight_hour_launcher_uses_separate_output_lineage(tmp_path):
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_mamba_video_only_3slot_h200_8h.sh"
    assert script.is_file()

    source = script.read_text()
    assert "#SBATCH --partition=gpu_h200" in source
    assert "#SBATCH --time=08:00:00" in source
    assert "#SBATCH --requeue" not in source

    (tmp_path / "configs/accelerate").mkdir(parents=True)
    (tmp_path / "configs/accelerate/h200_8gpu_2node.yaml").write_text("{}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mamba_prefix = tmp_path / "mamba"
    (mamba_prefix / "bin").mkdir(parents=True)
    call_log = tmp_path / "calls.txt"

    _write_executable(bin_dir / "scontrol", "#!/bin/bash\nprintf 'node-a\\n'\n")
    _write_executable(
        bin_dir / "getent", "#!/bin/bash\nprintf '10.0.0.1 STREAM node-a\\n'\n"
    )
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "ibstat", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "srun",
        "#!/bin/bash\nset -eu\n"
        "while (($#)); do\n"
        "  if [[ $1 == --export=* ]]; then\n"
        "    IFS=, read -ra exports <<< \"${1#--export=}\"\n"
        "    for entry in \"${exports[@]}\"; do [[ $entry == *=* ]] && export \"$entry\"; done\n"
        "  fi\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(mamba_prefix / "bin/python", "#!/bin/bash\nexit 0\n")
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_DIR": str(tmp_path),
            "MAMBA_ENV_PREFIX": str(mamba_prefix),
            "CALL_LOG": str(call_log),
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "node-a,node-b",
            "SLURM_NODEID": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert (
        f"NCCL logs: {tmp_path}/logs/nccl-joint-simgen-video-only-3slot-h200-8h-12345"
        in result.stdout
    )
    calls = call_log.read_text()
    assert "joint_simgen_video_only.py" in calls
    assert "logging.output_dir=outputs/joint_simgen_video_only_3slot_h200_8h" in calls
    assert "logging.run_name=joint-simgen-video-only-3slot-h200-8h" in calls
