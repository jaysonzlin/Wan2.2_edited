"""Configuration loading for the isolated native-Wan SimGen ablation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from training.pc_config import _apply_override


_PC_FIELD_NAMES = {
    "bca",
    "bridge",
    "bridges",
    "pc",
    "pc_type",
    "point_conditioning",
    "enable_pc_loss",
    "enable_rigid_loss",
    "enable_deform_loss",
    "utonia_feature_dim",
    "conditioning",
    "synthetic_object_slots",
    "num_points",
}


def _require_exact(mapping: dict, key: str, expected, message: str) -> None:
    if mapping.get(key) != expected:
        raise ValueError(message)


def _reject_pc_fields(config: dict) -> None:
    for section in ("data", "model", "objective", "optimizer", "training", "sampling"):
        fields = config.get(section, {})
        forbidden = _PC_FIELD_NAMES.intersection(fields)
        if forbidden:
            raise ValueError(f"vanilla-video configuration forbids PC fields: {sorted(forbidden)}")


def validate_simgen_vanilla_video_config(config: dict) -> None:
    """Reject settings that would break the controlled vanilla-Wan comparison."""
    _reject_pc_fields(config)
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    visualization = config.get("visualization", {})
    validation = config.get("validation", {})
    logging = config.get("logging", {})

    for key, expected in {
        "width": 480,
        "height": 480,
        "num_frames": 49,
        "train_start": 0,
        "train_end": 127,
        "validation_start": 490,
        "validation_end": 499,
        "train_batch_size": 1,
        "dataloader_num_workers": 4,
    }.items():
        message = "training split must be sample_0 through sample_127" if key == "train_end" else (
            "validation split must be sample_490 through sample_499" if key == "validation_end" else f"data.{key} must be {expected}"
        )
        _require_exact(data, key, expected, message)
    _require_exact(model, "checkpoint_dir", "Wan2.2-TI2V-5B", "model.checkpoint_dir must use Wan2.2-TI2V-5B")
    _require_exact(model, "gradient_checkpointing", True, "model.gradient_checkpointing must be true")
    _require_exact(model, "history_frames", 4, "model.history_frames must be 4")
    _require_exact(objective, "video_type", "flow", "objective.video_type must be 'flow'")
    _require_exact(objective, "num_train_timesteps", 1000, "objective.num_train_timesteps must be 1000")
    _require_exact(objective, "time_shift", 5.0, "objective.time_shift must be 5.0")
    if set(optimizer) != {"video"}:
        raise ValueError("vanilla-video configuration permits only optimizer.video")
    video_optimizer = optimizer["video"]
    if video_optimizer != {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1}:
        raise ValueError("optimizer.video must use the established Wan settings")
    for key, expected in {
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
    }.items():
        _require_exact(training, key, expected, f"training.{key} must be {expected!r}")
    _require_exact(sampling, "num_inference_steps", 50, "sampling.num_inference_steps must be 50")
    _require_exact(visualization, "every_steps", 1000, "visualization.every_steps must be 1000")
    _require_exact(visualization, "fps", 12, "visualization.fps must be 12")
    _require_exact(validation, "every_steps", 1000, "validation.every_steps must be 1000")
    _require_exact(logging, "output_dir", "outputs/simgen_vanilla_wan_video", "logging.output_dir must use the vanilla-video lineage")
    _require_exact(logging, "run_name", "simgen-vanilla-wan-video", "logging.run_name must use the vanilla-video lineage")


def load_simgen_vanilla_video_config(path: str | Path, overrides: list[str]) -> dict:
    """Load, override, and validate the dedicated vanilla-video YAML."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    config = deepcopy(config)
    for override in overrides:
        _apply_override(config, override)
    validate_simgen_vanilla_video_config(config)
    return config
