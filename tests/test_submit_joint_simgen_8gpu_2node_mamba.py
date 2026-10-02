import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_mamba_launcher_runs_standard_joint_training_without_singularity(tmp_path):
    """The Mamba variant must retain the standard joint-training launch contract."""
    script = PROJECT_DIR / "submit_joint_simgen_8gpu_2node_mamba.sh"
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
        "#!/bin/bash\n"
        "set -eu\n"
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
    _write_executable(mamba_prefix / "bin/python", "#!/bin/bash\nexit 0\n")
    _write_executable(
        mamba_prefix / "bin/accelerate",
        "#!/bin/bash\n"
        "printf 'accelerate %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "printf 'nccl_debug=%s distributed_debug=%s\\n' \"$NCCL_DEBUG\" \"$TORCH_DISTRIBUTED_DEBUG\" >> \"$CALL_LOG\"\n",
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
    calls = call_log.read_text()
    assert "nccl_debug=INFO distributed_debug=INFO" in calls
    assert (
        "accelerate launch --config_file configs/accelerate/h200_8gpu_2node.yaml "
        "--machine_rank 0 --main_process_ip 10.0.0.1 --main_process_port 32345 "
        "joint_simgen.py --config configs/train/joint_simgen_480_8gpu.yaml "
        "data.train_start=0 data.train_end=127 data.validation_start=490 "
        "data.validation_end=499 validation.every_steps=1000 "
        "visualization.every_steps=1000 training.resume_from_checkpoint=latest"
    ) in calls
