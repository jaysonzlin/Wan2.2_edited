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
    frozen_auxiliary = training.get("freeze_pc_and_bridges", False)
    freeze_auxiliary_pc_state = config.get("sampling", {}).get(
        "freeze_auxiliary_pc_state", False
    )
    if frozen_auxiliary is not freeze_auxiliary_pc_state:
        raise ValueError(
            "training.freeze_pc_and_bridges and sampling.freeze_auxiliary_pc_state must match"
        )
    if frozen_auxiliary and training.get("resume_from_checkpoint") is not None:
        raise ValueError("frozen-auxiliary video-only training must not resume")
    if not frozen_auxiliary and training.get("resume_from_checkpoint") not in (None, "latest"):
        raise ValueError(
            "training.resume_from_checkpoint must be null or latest for the isolated run"
        )
    allowed_steps = (4000,) if frozen_auxiliary else (5000, 100000)
    if training.get("max_train_steps") not in allowed_steps:
        raise ValueError("training.max_train_steps must be 5000 or 100000")
    if training.get("checkpoint_every_steps") != 1000:
        raise ValueError("training.checkpoint_every_steps must be 1000")
    if config.get("validation", {}).get("every_steps") != 1000:
        raise ValueError("validation.every_steps must be 1000")
    if config.get("visualization", {}).get("every_steps") != 1000:
        raise ValueError("visualization.every_steps must be 1000")
    if config.get("sampling", {}).get("num_inference_steps") != 50:
        raise ValueError("sampling.num_inference_steps must be 50")
    logging = config.get("logging", {})
    output_dir = logging.get("output_dir")
    run_name = logging.get("run_name")
    approved_lineage_pairs = {
        ("outputs/joint_simgen_video_only_3slot", "joint-simgen-video-only-3slot"),
        (
            "outputs/joint_simgen_video_only_3slot_h200_8h",
            "joint-simgen-video-only-3slot-h200-8h",
        ),
        (
            "outputs/joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k",
            "joint-simgen-video-only-3slot-frozen-auxiliary-h200-4k",
        ),
    }
    if output_dir not in {pair[0] for pair in approved_lineage_pairs}:
        raise ValueError("logging.output_dir must use the isolated video-only lineage")
    if run_name not in {pair[1] for pair in approved_lineage_pairs}:
        raise ValueError("logging.run_name must use the isolated video-only lineage")
    if (output_dir, run_name) not in approved_lineage_pairs:
        raise ValueError("logging.output_dir and logging.run_name must use an approved lineage pair")
    frozen_pair = (
        "outputs/joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k",
        "joint-simgen-video-only-3slot-frozen-auxiliary-h200-4k",
    )
    if ((output_dir, run_name) == frozen_pair) is not frozen_auxiliary:
        raise ValueError("frozen-auxiliary lineage must match frozen configuration")


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
