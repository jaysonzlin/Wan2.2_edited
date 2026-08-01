"""Configuration loading and validation for joint Wan--PhysCtrl training."""

from copy import deepcopy
from pathlib import Path

import yaml

from training.pc_config import _apply_override


def load_joint_config(path: str | Path, overrides: list[str]) -> dict:
    """Load a joint-training YAML file, apply dotted overrides, and validate it."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    config = deepcopy(config)
    for override in overrides:
        _apply_override(config, override)
    validate_joint_config(config)
    return config


def validate_joint_config(config: dict) -> None:
    """Reject configuration values outside the initial joint-training contract."""
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    optimizer = config.get("optimizer", {})
    if data.get("num_frames") != 49:
        raise ValueError("data.num_frames must be 49")
    if data.get("num_points") != 2048:
        raise ValueError("data.num_points must be 2048")
    if data.get("train_batch_size") != 1:
        raise ValueError("data.train_batch_size must be 1")
    if not isinstance(data.get("max_objects_per_sample"), int) or data["max_objects_per_sample"] <= 0:
        raise ValueError("data.max_objects_per_sample must be a positive integer")
    if (model.get("wan_cross_attention_blocks"), model.get("physctrl_blocks")) != (8, 8):
        raise ValueError("model must pair 8 Wan blocks with 8 PhysCtrl blocks")
    if model.get("interaction_dim") != 512:
        raise ValueError("model.interaction_dim must be 512")
    if model.get("interaction_heads") != 8:
        raise ValueError("model.interaction_heads must be 8")
    if objective.get("pc_type") != "ddpm_x0":
        raise ValueError("objective.pc_type must be 'ddpm_x0'")
    if objective.get("text_dropout_probability") != 0:
        raise ValueError("objective.text_dropout_probability must be 0")
    expected_optimizer = {
        "lr": 1.0e-5,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "weight_decay": 0.1,
    }
    for key, value in expected_optimizer.items():
        if optimizer.get(key) != value:
            raise ValueError(f"optimizer.{key} must be {value!r}")
    training = config.get("training", {})
    if not isinstance(training.get("denoised_latent_mse_every_steps"), int) or training["denoised_latent_mse_every_steps"] <= 0:
        raise ValueError("training.denoised_latent_mse_every_steps must be a positive integer")
    if not isinstance(training.get("checkpoints_total_limit"), int) or training["checkpoints_total_limit"] <= 0:
        raise ValueError("training.checkpoints_total_limit must be a positive integer")
    resume = training.get("resume_from_checkpoint")
    if resume is not None and not isinstance(resume, str):
        raise ValueError("training.resume_from_checkpoint must be null, 'latest', or a path string")
