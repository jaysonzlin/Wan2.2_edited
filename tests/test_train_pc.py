import subprocess
import sys
from pathlib import Path

from train_pc import visualization_path


def test_train_pc_help_is_local_only():
    result = subprocess.run(
        [sys.executable, "train_pc.py", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_visualization_path_is_inside_configured_vis_directory():
    assert visualization_path(Path("outputs/run"), "vis", 12) == Path("outputs/run/vis/epoch_0012.mp4")


def test_readme_documents_pc_flow_entrypoint():
    readme = Path("README.md").read_text()

    assert "train_pc.py --config configs/train/config_pc.yaml" in readme
    assert "pc.hdf5" in readme
