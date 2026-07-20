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
    flow = config.get("flow", {})
    if data.get("num_frames") != 49:
        raise ValueError("data.num_frames must be 49")
    if data.get("num_points") != 2048:
        raise ValueError("data.num_points must be 2048")
    if (model.get("n_layers"), model.get("latent_dim"), model.get("num_heads")) != (8, 256, 4):
        raise ValueError("model must be 8 layers, width 256, and 4 heads")
    if flow.get("prediction_type") != "flow":
        raise ValueError("flow.prediction_type must be 'flow'")
    if not isinstance(flow.get("time_shift"), (int, float)) or flow["time_shift"] <= 0:
        raise ValueError("flow.time_shift must be positive")
    if config.get("lr_scheduler") not in {"cosine", "constant"}:
        raise ValueError("lr_scheduler must be 'cosine' or 'constant'")
