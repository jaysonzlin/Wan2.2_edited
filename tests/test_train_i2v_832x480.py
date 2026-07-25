import os
import subprocess
import sys
from pathlib import Path

from training.overfit_config import load_config


def test_help_does_not_import_remote_only_training_dependencies():
    result = subprocess.run(
        [sys.executable, "train_i2v_832x480.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_dataset_is_explicitly_native_832_by_480():
    source = Path("train_i2v_832x480.py").read_text()

    assert "KubricI2VOverfitDataset(" in source
    assert "expected_size=(832, 480)" in source
    assert "from accelerate import Accelerator" in source
    assert "accelerator = Accelerator(" in source


def test_832x480_config_targets_native_resolution_dataset():
    config = load_config("configs/train/overfit_kubric_i2v_832x480.yaml", [])

    assert config["data"]["dataset_root"] == "training_dataset_832x480"
    assert config["data"]["width"] == 832
    assert config["data"]["height"] == 480
    assert config["data"]["num_frames"] == 49
