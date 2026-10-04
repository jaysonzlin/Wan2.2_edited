import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_validation_launcher_runs_non_frozen_renderer_with_requested_paths(tmp_path):
    """The Slurm wrapper supplies its checkpoint and destination to the renderer."""
    script = PROJECT_DIR / "submit_visualize_joint_simgen_validation.sh"
    checkpoint = tmp_path / "checkpoint-9000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_text("state\n")
    output_dir = tmp_path / "validation-videos"
    mamba_prefix = tmp_path / "mamba"
    python = mamba_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    _write_executable(
        python,
        "#!/bin/bash\nprintf 'python %s\\n' \"$*\" >> \"$CALL_LOG\"\n",
    )
    module_dir = tmp_path / "bin"
    module_dir.mkdir()
    _write_executable(module_dir / "module", "#!/bin/bash\nexit 0\n")

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{module_dir}:{os.environ['PATH']}",
            "PROJECT_DIR": str(PROJECT_DIR),
            "MAMBA_ENV_PREFIX": str(mamba_prefix),
            "CHECKPOINT_PATH": str(checkpoint),
            "OUTPUT_DIR": str(output_dir),
            "CALL_LOG": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"Checkpoint: {checkpoint}" in result.stdout
    assert f"Output: {output_dir}" in result.stdout
    assert calls.read_text() == (
        "python visualize_joint_simgen_validation.py "
        f"--checkpoint {checkpoint} --output-dir {output_dir}\n"
    )
