"""YAML configuration loading for the Kubric I2V overfit workflow."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str]) -> dict:
    """Load a YAML mapping, apply ``section.key=value`` overrides, and validate it."""
    config_path = Path(path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top-level configuration must be a mapping")

    merged = deepcopy(config)
    for override in overrides:
        _apply_override(merged, override)
    validate_config(merged)
    return merged


def _apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}; expected section.key=value")
    key, raw_value = override.split("=", 1)
    keys = key.split(".")
    if not key or any(not part for part in keys):
        raise ValueError(f"Invalid override key {key!r}")

    destination = config
    for part in keys[:-1]:
        existing = destination.get(part)
        if existing is None:
            destination[part] = {}
        elif not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested override below non-mapping key {part!r}")
        destination = destination[part]
    destination[keys[-1]] = yaml.safe_load(raw_value)


def validate_config(config: dict) -> None:
    """Reject configuration values that violate the fixed training contract."""
    data = config.get("data", {})
    if "num_frames" in data:
        num_frames = data["num_frames"]
        if not isinstance(num_frames, int) or num_frames < 5 or num_frames % 4 != 1:
            raise ValueError("data.num_frames must follow Wan's 4n + 1 temporal format")

    training = config.get("training", {})
    for key in ("max_train_steps", "warmup_steps", "checkpoint_every_steps"):
        if key in training and (
            not isinstance(training[key], int) or training[key] <= 0
        ):
            raise ValueError(f"training.{key} must be a positive integer")

    validation = config.get("validation", {})
    if validation.get("enabled"):
        raise ValueError("Validation is disabled for this intentional overfit run")
