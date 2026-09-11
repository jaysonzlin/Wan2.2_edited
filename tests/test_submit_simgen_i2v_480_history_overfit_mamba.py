import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_mamba_launcher_preflights_training_extensions_before_eight_gpu_launch(
    tmp_path: Path,
) -> None:
    """A bad Mamba training stack must fail before distributed training starts."""
    script = PROJECT_DIR / "submit_simgen_i2v_480_history_overfit_mamba.sh"
    (tmp_path / "configs/accelerate").mkdir(parents=True)
    (tmp_path / "configs/accelerate/h200_8gpu_2node.yaml").write_text("{}\n")
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
        bin_dir / "srun",
        "#!/bin/bash\n"
        "printf 'srun %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "while (($#)); do\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(
        mamba_prefix / "bin/python",
        "#!/bin/bash\n"
        "printf 'python %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "printf 'MAMBA_PREFLIGHT_OK cuda_devices=4\\n'\n",
    )
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\n"
        "printf 'accelerate %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
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
    assert "MAMBA_PREFLIGHT_OK cuda_devices=4" in result.stdout
    calls = call_log.read_text()
    assert "python -c import torch; import flash_attn; import spconv.pytorch; import torch_scatter;" in calls
    assert "accelerate launch --config_file configs/accelerate/h200_8gpu_2node.yaml" in calls
    assert "--machine_rank 0 --main_process_ip 10.0.0.1 --main_process_port 32345" in calls
    assert calls.count("srun ") == 2
    launcher = script.read_text()
    assert "#SBATCH --constraint=h200&holyndr" in launcher
    assert "#SBATCH --exclude=holygpu8a12204" in launcher
    assert "#SBATCH --switches=1" in launcher
    assert "#SBATCH --gres=gpu:nvidia_h200:4" in launcher
    assert "singularity" not in launcher.lower()


def test_mamba_launcher_stops_before_accelerate_when_preflight_fails(
    tmp_path: Path,
) -> None:
    """A failed node preflight must prevent the separate training srun step."""
    script = PROJECT_DIR / "submit_simgen_i2v_480_history_overfit_mamba.sh"
    (tmp_path / "configs/accelerate").mkdir(parents=True)
    (tmp_path / "configs/accelerate/h200_8gpu_2node.yaml").write_text("{}\n")
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
        bin_dir / "srun",
        "#!/bin/bash\n"
        "printf 'srun %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "while (($#)); do\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(
        mamba_prefix / "bin/python",
        "#!/bin/bash\n"
        "printf 'python %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "exit 42\n",
    )
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\nprintf 'accelerate %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
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

    assert result.returncode == 42
    calls = call_log.read_text()
    assert calls.count("srun ") == 1
    assert "accelerate " not in calls
