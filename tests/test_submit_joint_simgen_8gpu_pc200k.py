import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_joint_pc200k_launcher_runs_the_mamba_accelerate_environment_without_preflight(
    tmp_path: Path,
) -> None:
    """The joint job must launch through Mamba once, without a diagnostic srun."""
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_pc200k.sh"
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
        bin_dir / "module",
        "#!/bin/bash\nprintf 'module %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )
    _write_executable(
        bin_dir / "srun",
        "#!/bin/bash\n"
        "printf 'srun %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "while (($#)); do\n"
        "  if [[ $1 == bash ]]; then shift; exec bash \"$@\"; fi\n"
        "  shift\n"
        "done\n",
    )
    _write_executable(mamba_prefix / "bin/python", "#!/bin/bash\nexit 1\n")
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
            "RESUME_FROM_CHECKPOINT": "",
        },
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text()
    assert calls.count("srun ") == 1
    assert "module load Mambaforge" in calls
    assert "module load cuda/12.4.1" in calls
    assert "module load gcc/9.5.0-fasrc01" in calls
    assert (
        "accelerate launch --config_file configs/accelerate/h200_8gpu_2node.yaml "
        "--machine_rank 0 --main_process_ip 10.0.0.1 --main_process_port 32345 "
        "joint_simgen.py --config configs/train/joint_simgen_480_8gpu_pc200k.yaml"
    ) in calls

    source = script.read_text()
    assert "#SBATCH --constraint=h200&holyndr" in source
    assert "#SBATCH --exclude=holygpu8a12204,holygpu8a18103" in source
    assert "#SBATCH --switches=1" in source
    assert "#SBATCH --gres=gpu:nvidia_h200:4" in source
