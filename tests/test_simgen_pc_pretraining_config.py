from pathlib import Path

import pytest
import yaml

from training.simgen_pc_pretraining_config import load_simgen_pc_pretraining_config


CONFIG_PATH = "configs/train/pretrain_simgen_pc_480_4gpu.yaml"


def test_pretraining_config_loads_the_fixed_200k_experiment():
    config = load_simgen_pc_pretraining_config(CONFIG_PATH, [])

    assert config["data"]["train_start"] == 0
    assert config["data"]["train_end"] == 127
    assert config["data"]["validation_start"] == 490
    assert config["data"]["validation_end"] == 499
    assert config["training"]["max_train_steps"] == 200_000
    assert config["training"]["checkpoint_every_steps"] == 250
    assert config["training"]["checkpoints_total_limit"] == 2
    assert config["training"]["resume_from_checkpoint"] is None
    assert config["validation"]["every_steps"] == 1000
    assert config["visualization"]["every_steps"] == 1000


def test_pretraining_config_rejects_non_joint_compatible_history(tmp_path):
    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    config["model"]["history_frames"] = 1
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="history_frames must be 4"):
        load_simgen_pc_pretraining_config(invalid_path, [])


def test_pretraining_config_rejects_more_than_two_resumable_checkpoints(tmp_path):
    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    config["training"]["checkpoints_total_limit"] = 3
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="checkpoints_total_limit must be 2"):
        load_simgen_pc_pretraining_config(invalid_path, [])
