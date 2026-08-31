import pytest

from training.simgen_joint_config import validate_simgen_joint_config


def test_simgen_config_requires_480_history_flow_and_ddpm():
    config = {
        "data": {"width": 480, "height": 480, "num_frames": 49, "num_points": 2048},
        "model": {"conditioning": "history", "history_frames": 4},
        "objective": {
            "video_type": "flow", "pc_type": "ddpm_x0",
            "enable_rigid_loss": False, "enable_deform_loss": False,
        },
    }
    config["data"].update(
        train_start=0, train_end=489, validation_start=490, validation_end=499
    )
    config["validation"] = {"every_steps": 250}
    config["visualization"] = {"every_steps": 250}
    config["optimizer"] = {
        "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
    }

    validate_simgen_joint_config(config)

    config["visualization"]["every_steps"] = 1000
    validate_simgen_joint_config(config)

    config["validation"]["every_steps"] = 1000
    validate_simgen_joint_config(config)

    config["data"]["train_end"] = 127
    validate_simgen_joint_config(config)

    config["data"]["width"] = 832
    with pytest.raises(ValueError, match="must both be 480"):
        validate_simgen_joint_config(config)


def test_simgen_config_requires_exact_splits_cadences_and_optimizer_groups():
    config = {
        "data": {
            "width": 480, "height": 480, "num_frames": 49, "num_points": 2048,
            "train_start": 1, "train_end": 489, "validation_start": 490, "validation_end": 499,
        },
        "model": {"conditioning": "history", "history_frames": 4},
        "objective": {
            "video_type": "flow", "pc_type": "ddpm_x0",
            "enable_rigid_loss": False, "enable_deform_loss": False,
        },
        "validation": {"every_steps": 250},
        "visualization": {"every_steps": 250},
        "optimizer": {
            "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        },
    }

    with pytest.raises(ValueError, match="training split must start at sample_0"):
        validate_simgen_joint_config(config)


def test_simgen_config_requires_history_conditioning():
    config = {
        "data": {
            "width": 480, "height": 480, "num_frames": 49, "num_points": 2048,
            "train_start": 0, "train_end": 489, "validation_start": 490, "validation_end": 499,
        },
        "model": {"conditioning": "velocity", "history_frames": 4},
        "objective": {
            "video_type": "flow", "pc_type": "ddpm_x0",
            "enable_rigid_loss": False, "enable_deform_loss": False,
        },
        "validation": {"every_steps": 250},
        "visualization": {"every_steps": 250},
        "optimizer": {
            "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        },
    }

    with pytest.raises(ValueError, match="model.conditioning must be 'history'"):
        validate_simgen_joint_config(config)
