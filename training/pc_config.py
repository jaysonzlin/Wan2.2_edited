"""Configuration loading and validation for point-cloud flow training."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_pc_config(path: str | Path, overrides: list[str]) -> dict:
    """Load a PC YAML config, apply dotted overrides, and validate it."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    config = deepcopy(config)
    for override in overrides:
        _apply_override(config, override)
    validate_pc_config(config)
    return config


def _apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}; expected section.key=value")
    raw_key, raw_value = override.split("=", 1)
    keys = raw_key.split(".")
    if not raw_key or any(not key for key in keys):
        raise ValueError(f"Invalid override key {raw_key!r}")
    destination = config
    for key in keys[:-1]:
        value = destination.get(key)
        if value is None:
            destination[key] = {}
        elif not isinstance(value, dict):
            raise ValueError(f"Cannot set nested override below non-mapping key {key!r}")
        destination = destination[key]
    destination[keys[-1]] = yaml.safe_load(raw_value)


def validate_pc_config(config: dict) -> None:
    """Reject values that violate the fixed PC architecture contract."""
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    if data.get("num_frames") != 49:
        raise ValueError("data.num_frames must be 49")
    if data.get("num_points") != 2048:
        raise ValueError("data.num_points must be 2048")
    if model.get("num_heads") != model.get("latent_dim", 0) // 64:
        raise ValueError("model.num_heads must equal model.latent_dim // 64")
    if (model.get("n_layers"), model.get("latent_dim"), model.get("num_heads")) != (8, 256, 4):
        raise ValueError("model must be 8 layers, width 256, and 4 heads")
    if model.get("point_embed") is not True:
        raise ValueError("model.point_embed must be true")
    if model.get("frame_cond") is not True:
        raise ValueError("model.frame_cond must be true")
    if model.get("transformer_block") != "SpatialTemporalTransformerBlock":
        raise ValueError(
            "model.transformer_block must be 'SpatialTemporalTransformerBlock'"
        )
    if objective.get("type") not in {"flow", "ddpm"}:
        raise ValueError("objective.type must be 'flow' or 'ddpm'")
    if not isinstance(objective.get("num_train_timesteps"), int) or objective["num_train_timesteps"] <= 0:
        raise ValueError("objective.num_train_timesteps must be positive")
    if not isinstance(objective.get("time_shift"), (int, float)) or objective["time_shift"] <= 0:
        raise ValueError("objective.time_shift must be positive")
    if objective["type"] == "ddpm" and objective.get("beta_schedule") != "linear":
        raise ValueError("DDPM objective.beta_schedule must be 'linear'")
    if config.get("lr_scheduler") not in {"cosine", "constant"}:
        raise ValueError("lr_scheduler must be 'cosine' or 'constant'")
