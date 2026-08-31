"""Validation for the fixed-contract joint SimGen experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

from training.pc_config import _apply_override


def validate_simgen_joint_config(config: dict) -> None:
    data, model, objective = (config.get(key, {}) for key in ("data", "model", "objective"))
    training = config.get("training", {})
    if (data.get("width"), data.get("height")) != (480, 480):
        raise ValueError("data.width and data.height must both be 480")
    if data.get("num_frames") != 49 or data.get("num_points") != 2048:
        raise ValueError("data.num_frames and data.num_points must be 49 and 2048")
    if model.get("history_frames") != 4:
        raise ValueError("model.history_frames must be 4")
    if model.get("conditioning") != "history":
        raise ValueError("model.conditioning must be 'history'")
    if objective.get("video_type") != "flow":
        raise ValueError("objective.video_type must be 'flow'")
    if objective.get("pc_type") != "ddpm_x0":
        raise ValueError("objective.pc_type must be 'ddpm_x0'")
    if objective.get("enable_rigid_loss") is not False:
        raise ValueError("objective.enable_rigid_loss must be false")
    if objective.get("enable_deform_loss") is not False:
        raise ValueError("objective.enable_deform_loss must be false")
    train_end = data.get("train_end")
    if (
        data.get("train_start") != 0
        or not isinstance(train_end, int)
        or isinstance(train_end, bool)
        or not 0 <= train_end <= 489
    ):
        raise ValueError("training split must start at sample_0 and end no later than sample_489")
    expected_validation_range = {"validation_start": 490, "validation_end": 499}
    if any(data.get(key) != value for key, value in expected_validation_range.items()):
        raise ValueError("validation split must be sample_490 through sample_499")
    if config.get("validation", {}).get("every_steps") not in {250, 1000}:
        raise ValueError("validation.every_steps must be 250 or 1000")
    if config.get("visualization", {}).get("every_steps") not in {250, 1000}:
        raise ValueError("visualization.every_steps must be 250 or 1000")
    expected_optimizer = {
        "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
    }
    if config.get("optimizer") != expected_optimizer:
        raise ValueError("optimizer must use the established joint AdamW groups")
    pretrained_pc_weights = training.get("pretrained_pc_weights")
    if pretrained_pc_weights is not None and (
        not isinstance(pretrained_pc_weights, str) or not pretrained_pc_weights.strip()
    ):
        raise ValueError("training.pretrained_pc_weights must be null or a non-empty path string")
    if pretrained_pc_weights and training.get("resume_from_checkpoint"):
        raise ValueError(
            "training.pretrained_pc_weights and training.resume_from_checkpoint cannot both be set"
        )


def load_simgen_joint_config(path: str | Path, overrides: list[str]) -> dict:
    with Path(path).open() as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    for override in overrides:
        _apply_override(config, override)
    validate_simgen_joint_config(config)
    return config
