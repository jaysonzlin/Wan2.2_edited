import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_capture_script_exports_sorted_sif_conda_and_pip_locks(tmp_path: Path) -> None:
    """A captured lock must come from the SIF's app environment, not the host."""
    script = PROJECT_DIR / "capture_cur_sif_lock.sh"
    image = tmp_path / "cur.sif"
    image.write_bytes(b"synthetic SIF")
    lock_dir = tmp_path / "locks"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "singularity",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "[[ $1 == exec && $2 == --nv && $3 == \"$SIF_IMAGE\" ]]\n"
        "case \"$4 $5 ${6:-} ${7:-}\" in\n"
        "  */conda\\ list\\ -p\\ /opt/conda/envs/app) printf '%s\\n' '# explicit' 'https://conda.example/python-3.10.1.conda' ;;\n"
        "  */python\\ -m\\ pip\\ freeze) printf '%s\\n' 'zeta==1' 'alpha==2' ;;\n"
        "  */python\\ --version\\ *) printf '%s\\n' 'Python 3.10.1' ;;\n"
        "  *) echo \"unexpected singularity invocation: $*\" >&2; exit 1 ;;\n"
        "esac\n",
    )

    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SIF_IMAGE": str(image),
            "SIF_LOCK_DIR": str(lock_dir),
        },
    )

    assert result.returncode == 0, result.stderr
    assert (lock_dir / "conda-explicit.txt").read_text() == (
        "# explicit\nhttps://conda.example/python-3.10.1.conda\n"
    )
    assert (lock_dir / "pip-freeze.txt").read_text() == (
        "--extra-index-url https://download.pytorch.org/whl/cu124\n"
        "--find-links https://data.pyg.org/whl/torch-2.4.0+cu124.html\n"
        "alpha==2\nzeta==1\n"
    )
    provenance = (lock_dir / "provenance.txt").read_text()
    assert "image_sha256=" in provenance
    assert "python_version=Python 3.10.1" in provenance


def test_lock_mode_recreates_prefix_from_captured_conda_and_pip_locks(
    tmp_path: Path,
) -> None:
    """Lock mode must not resolve the broad requirements or retain a hybrid prefix."""
    script = PROJECT_DIR / "create_mamba_env.sh"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    conda_lock = lock_dir / "conda-explicit.txt"
    pip_lock = lock_dir / "pip-freeze.txt"
    conda_lock.write_text("# explicit\nhttps://conda.example/python-3.10.1.conda\n")
    pip_lock.write_text("alpha==2\nzeta==1\n")
    env_prefix = tmp_path / "environment"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.txt"
    pip_freeze_output = tmp_path / "pip-freeze-output.txt"
    pip_freeze_output.write_text("alpha==2\nzeta==1\n")
    _write_executable(
        bin_dir / "mamba",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf 'mamba %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "prefix=\n"
        "while (($#)); do\n"
        "  if [[ $1 == --prefix ]]; then prefix=$2; shift 2; continue; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$prefix/bin\"\n"
        "cat > \"$prefix/bin/python\" <<'PYTHON'\n"
        "#!/bin/bash\n"
        "printf 'python %s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "if [[ $* == *'pip freeze --all'* ]]; then cat \"$PIP_FREEZE_OUTPUT\"; fi\n"
        "PYTHON\n"
        "chmod +x \"$prefix/bin/python\"\n",
    )

    result = subprocess.run(
        ["bash", script, "--from-sif-lock", "--recreate"],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MAMBA_EXE": str(bin_dir / "mamba"),
            "MAMBA_ENV_PREFIX": str(env_prefix),
            "MAMBA_PKGS_DIRS": str(tmp_path / "packages"),
            "SIF_LOCK_DIR": str(lock_dir),
            "CALL_LOG": str(call_log),
            "PIP_FREEZE_OUTPUT": str(pip_freeze_output),
        },
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text()
    assert f"mamba create --yes --prefix {env_prefix} --file {conda_lock}" in calls
    assert f"python -m pip install --no-user --no-cache-dir --requirement {pip_lock}" in calls
    assert "upgrade-strategy eager" not in calls
