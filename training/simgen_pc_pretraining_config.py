"""Configuration validation for joint-compatible SimGen PC pretraining."""

from __future__ import annotations

from pathlib import Path

import yaml

from training.pc_config import _apply_override


def _require_exact(value, expected, message: str) -> None:
    if value != expected:
        raise ValueError(message)


def validate_simgen_pc_pretraining_config(config: dict) -> None:
    """Reject runs that would not produce a joint-compatible PC initializer."""
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    training = config.get("training", {})
    if (data.get("width"), data.get("height")) != (480, 480):
        raise ValueError("data.width and data.height must both be 480")
    if data.get("num_frames") != 49 or data.get("num_points") != 2048:
        raise ValueError("data.num_frames and data.num_points must be 49 and 2048")
    _require_exact(data.get("train_start"), 0, "training split must start at sample_0")
    _require_exact(data.get("train_end"), 127, "training split must end at sample_127")
    if (data.get("validation_start"), data.get("validation_end")) != (490, 499):
        raise ValueError("validation split must be sample_490 through sample_499")
    _require_exact(data.get("train_batch_size"), 1, "data.train_batch_size must be 1")
    if not isinstance(data.get("utonia_cache_root"), str) or not data["utonia_cache_root"].strip():
        raise ValueError("data.utonia_cache_root must be a non-empty path string")
    if (model.get("n_layers"), model.get("latent_dim"), model.get("num_heads")) != (8, 256, 4):
        raise ValueError("model must be 8 layers, width 256, and 4 heads")
    if model.get("conditioning") != "history":
        raise ValueError("model.conditioning must be 'history'")
    _require_exact(model.get("history_frames"), 4, "model.history_frames must be 4")
    if model.get("point_embed") is not True:
        raise ValueError("model.point_embed must be true")
    if objective.get("type") != "ddpm" or objective.get("prediction_type") != "x0":
        raise ValueError("objective must be DDPM x0")
    _require_exact(objective.get("num_train_timesteps"), 1000, "objective.num_train_timesteps must be 1000")
    if objective.get("beta_schedule") != "linear":
        raise ValueError("objective.beta_schedule must be 'linear'")
    _require_exact(training.get("max_train_steps"), 200_000, "training.max_train_steps must be 200000")
    _require_exact(training.get("checkpoint_every_steps"), 250, "training.checkpoint_every_steps must be 250")
    _require_exact(training.get("checkpoints_total_limit"), 2, "training.checkpoints_total_limit must be 2")
    resume = training.get("resume_from_checkpoint")
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("training.resume_from_checkpoint must be null or a non-empty path string")
    if config.get("validation", {}).get("every_steps") != 1000:
        raise ValueError("validation.every_steps must be 1000")
    if config.get("visualization", {}).get("every_steps") != 1000:
        raise ValueError("visualization.every_steps must be 1000")
    expected_optimizer = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
    }
    if config.get("optimizer") != expected_optimizer:
        raise ValueError("optimizer must use the established joint PC AdamW settings")


def load_simgen_pc_pretraining_config(path: str | Path, overrides: list[str]) -> dict:
    """Load the fixed SimGen PC pretraining experiment and apply dotted overrides."""
    with Path(path).open() as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    for override in overrides:
        _apply_override(config, override)
    validate_simgen_pc_pretraining_config(config)
    return config
