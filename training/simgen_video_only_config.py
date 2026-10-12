"""Validation for the strict three-slot video-only SimGen ablation."""

from __future__ import annotations

from pathlib import Path

import yaml

from training.pc_config import _apply_override
from training.simgen_video_only_conditioning import SYNTHETIC_OBJECT_SLOTS


def validate_simgen_video_only_config(config: dict) -> None:
    """Reject any configuration that could reintroduce PC-side information."""
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    training = config.get("training", {})
    if (data.get("width"), data.get("height")) != (480, 480):
        raise ValueError("data.width and data.height must both be 480")
    if data.get("num_frames") != 49 or data.get("num_points") != 2048:
        raise ValueError("data.num_frames and data.num_points must be 49 and 2048")
    if data.get("train_batch_size") != 1:
        raise ValueError("data.train_batch_size must be 1")
    if data.get("synthetic_object_slots") != SYNTHETIC_OBJECT_SLOTS:
        raise ValueError("data.synthetic_object_slots must be 3")
    if "utonia_cache_root" in data:
        raise ValueError("data.utonia_cache_root is forbidden for video-only training")
    if (data.get("train_start"), data.get("train_end")) != (0, 127):
        raise ValueError("training split must be sample_0 through sample_127")
    if (data.get("validation_start"), data.get("validation_end")) != (490, 499):
        raise ValueError("validation split must be sample_490 through sample_499")
    if model.get("conditioning") != "history" or model.get("history_frames") != 4:
        raise ValueError("model must use four-frame history conditioning")
    feature_dim = model.get("utonia_feature_dim")
    if (
        not isinstance(feature_dim, int)
        or isinstance(feature_dim, bool)
        or feature_dim <= 0
    ):
        raise ValueError("model.utonia_feature_dim must be a positive integer")
    if objective.get("video_type") != "flow":
        raise ValueError("objective.video_type must be 'flow'")
    if objective.get("pc_type") != "ddpm_x0":
        raise ValueError("objective.pc_type must be 'ddpm_x0'")
    if objective.get("point_conditioning") != "noise_video_only":
        raise ValueError("objective.point_conditioning must be 'noise_video_only'")
    if objective.get("enable_pc_loss") is not False:
        raise ValueError("objective.enable_pc_loss must be false")
    if objective.get("enable_rigid_loss") is not False:
        raise ValueError("objective.enable_rigid_loss must be false")
    if objective.get("enable_deform_loss") is not False:
        raise ValueError("objective.enable_deform_loss must be false")
    if objective.get("num_train_timesteps") != 1000:
        raise ValueError("objective.num_train_timesteps must be 1000")
    if objective.get("beta_schedule") != "linear" or objective.get("time_shift") != 5.0:
        raise ValueError("objective must use the established DDPM and flow settings")
    expected_optimizer = {
        "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
    }
    if config.get("optimizer") != expected_optimizer:
        raise ValueError("optimizer must use the established joint AdamW groups")
    if training.get("pretrained_pc_weights") is not None:
        raise ValueError("training.pretrained_pc_weights must be null")
    if training.get("resume_from_checkpoint") not in (None, "latest"):
        raise ValueError(
            "training.resume_from_checkpoint must be null or latest for the isolated run"
        )
    if training.get("max_train_steps") != 100000:
        raise ValueError("training.max_train_steps must be 100000")
    if training.get("checkpoint_every_steps") != 1000:
        raise ValueError("training.checkpoint_every_steps must be 1000")
    if config.get("validation", {}).get("every_steps") != 1000:
        raise ValueError("validation.every_steps must be 1000")
    if config.get("visualization", {}).get("every_steps") != 1000:
        raise ValueError("visualization.every_steps must be 1000")
    if config.get("sampling", {}).get("num_inference_steps") != 50:
        raise ValueError("sampling.num_inference_steps must be 50")
    logging = config.get("logging", {})
    if logging.get("output_dir") != "outputs/joint_simgen_video_only_3slot":
        raise ValueError("logging.output_dir must use the isolated video-only lineage")
    if logging.get("run_name") != "joint-simgen-video-only-3slot":
        raise ValueError("logging.run_name must use the isolated video-only lineage")


def load_simgen_video_only_config(path: str | Path, overrides: list[str]) -> dict:
    """Load and validate the isolated video-only experiment configuration."""
    with Path(path).open() as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level configuration must be a mapping")
    for override in overrides:
        _apply_override(config, override)
    validate_simgen_video_only_config(config)
    return config
