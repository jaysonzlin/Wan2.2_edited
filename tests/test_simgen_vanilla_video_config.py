from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from training.simgen_vanilla_video_config import (
    load_simgen_vanilla_video_config,
    validate_simgen_vanilla_video_config,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _config():
    return {
        "data": {
            "dataset_root": "../simgen/runs/panda_ball_can",
            "width": 480,
            "height": 480,
            "num_frames": 49,
            "train_start": 0,
            "train_end": 127,
            "validation_start": 490,
            "validation_end": 499,
            "train_batch_size": 1,
            "dataloader_num_workers": 4,
        },
        "model": {
            "checkpoint_dir": "Wan2.2-TI2V-5B",
            "gradient_checkpointing": True,
            "history_frames": 4,
        },
        "objective": {
            "video_type": "flow",
            "num_train_timesteps": 1000,
            "time_shift": 5.0,
        },
        "optimizer": {
            "video": {
                "lr": 1.0e-5,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.1,
            }
        },
        "training": {
            "seed": 42,
            "gradient_accumulation_steps": 1,
            "max_train_steps": 100000,
            "warmup_steps": 200,
            "lr_scheduler": "constant",
            "max_grad_norm": 2.0,
            "mixed_precision": "bf16",
            "checkpoint_every_steps": 1000,
            "checkpoints_total_limit": 2,
            "resume_from_checkpoint": None,
        },
        "sampling": {"num_inference_steps": 50},
        "visualization": {"every_steps": 1000, "fps": 12},
        "validation": {"every_steps": 1000},
        "logging": {
            "output_dir": "outputs/simgen_vanilla_wan_video",
            "report_to": "wandb",
            "project": "joint-simgen",
            "run_name": "simgen-vanilla-wan-video",
        },
    }


def test_committed_vanilla_config_loads_with_the_controlled_contract():
    path = PROJECT_DIR / "configs/train/simgen_480_8gpu_vanilla_wan_video.yaml"

    config = load_simgen_vanilla_video_config(path, [])

    assert config == _config()


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training", "resume_from_checkpoint", "latest", "resume_from_checkpoint"),
        ("data", "train_end", 126, "training split"),
        ("validation", "every_steps", 500, "validation.every_steps"),
        ("optimizer", "bca", {}, "forbids PC"),
        ("objective", "enable_pc_loss", False, "forbids PC"),
        ("model", "utonia_feature_dim", 1088, "forbids PC"),
    ],
)
def test_vanilla_config_rejects_joint_or_uncontrolled_fields(section, key, value, message):
    config = deepcopy(_config())
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_simgen_vanilla_video_config(config)


def test_vanilla_loader_applies_safe_override_then_rejects_resume(tmp_path):
    path = tmp_path / "vanilla.yaml"
    path.write_text(yaml.safe_dump(_config()))

    loaded = load_simgen_vanilla_video_config(path, ["logging.project=test-project"])

    assert loaded["logging"]["project"] == "test-project"
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        load_simgen_vanilla_video_config(path, ["training.resume_from_checkpoint=latest"])
