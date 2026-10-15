from copy import deepcopy

import pytest
import yaml

from training.simgen_video_only_config import (
    load_simgen_video_only_config,
    validate_simgen_video_only_config,
)


def _config() -> dict:
    return {
        "data": {
            "dataset_root": "simgen",
            "width": 480,
            "height": 480,
            "num_frames": 49,
            "num_points": 2048,
            "train_start": 0,
            "train_end": 127,
            "validation_start": 490,
            "validation_end": 499,
            "synthetic_object_slots": 3,
            "train_batch_size": 1,
            "dataloader_num_workers": 4,
        },
        "model": {
            "conditioning": "history",
            "history_frames": 4,
            "checkpoint_dir": "Wan2.2-TI2V-5B",
            "gradient_checkpointing": True,
            "utonia_feature_dim": 1088,
        },
        "objective": {
            "video_type": "flow",
            "pc_type": "ddpm_x0",
            "point_conditioning": "noise_video_only",
            "enable_pc_loss": False,
            "enable_rigid_loss": False,
            "enable_deform_loss": False,
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "time_shift": 5.0,
        },
        "optimizer": {
            "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
            "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
        },
        "training": {
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
            "pretrained_pc_weights": None,
        },
        "sampling": {"num_inference_steps": 50},
        "visualization": {"every_steps": 1000, "fps": 12},
        "validation": {"every_steps": 1000},
        "logging": {
            "output_dir": "outputs/joint_simgen_video_only_3slot",
            "report_to": "wandb",
            "project": "joint-simgen",
            "run_name": "joint-simgen-video-only-3slot",
        },
    }


def test_video_only_config_loads_with_explicit_feature_width(tmp_path):
    path = tmp_path / "video_only.yaml"
    path.write_text(yaml.safe_dump(_config()))

    assert load_simgen_video_only_config(path, []) == _config()


def test_video_only_config_accepts_the_h200_eight_hour_lineage_pair():
    config = _config()
    config["logging"].update(
        output_dir="outputs/joint_simgen_video_only_3slot_h200_8h",
        run_name="joint-simgen-video-only-3slot-h200-8h",
    )

    validate_simgen_video_only_config(config)


def test_video_only_config_accepts_the_five_thousand_step_h200_run():
    config = _config()
    config["training"]["max_train_steps"] = 5000
    config["logging"].update(
        output_dir="outputs/joint_simgen_video_only_3slot_h200_8h",
        run_name="joint-simgen-video-only-3slot-h200-8h",
    )

    validate_simgen_video_only_config(config)


def test_video_only_config_rejects_an_unapproved_lineage_pair():
    config = _config()
    config["logging"].update(
        output_dir="outputs/joint_simgen_video_only_3slot_h200_8h",
        run_name="joint-simgen-video-only-3slot",
    )

    with pytest.raises(ValueError, match="approved lineage pair"):
        validate_simgen_video_only_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["data"].update(synthetic_object_slots=2), "synthetic_object_slots"),
        (lambda config: config["model"].pop("utonia_feature_dim"), "utonia_feature_dim"),
        (lambda config: config["model"].update(utonia_feature_dim=0), "utonia_feature_dim"),
        (lambda config: config["data"].update(utonia_cache_root="cache"), "utonia_cache_root"),
        (lambda config: config["objective"].update(point_conditioning="history"), "point_conditioning"),
        (lambda config: config["objective"].update(enable_pc_loss=True), "enable_pc_loss"),
        (lambda config: config["training"].update(pretrained_pc_weights="pc.safetensors"), "pretrained_pc_weights"),
        (lambda config: config["training"].update(resume_from_checkpoint="outputs/joint_simgen_8gpu/checkpoint-1000"), "resume_from_checkpoint"),
        (lambda config: config["sampling"].update(num_inference_steps=49), "num_inference_steps"),
        (lambda config: config["logging"].update(output_dir="outputs/joint_simgen_8gpu"), "output_dir"),
        (lambda config: config["logging"].update(run_name="joint-simgen"), "run_name"),
    ],
)
def test_video_only_config_rejects_non_isolated_point_conditioning(mutate, message):
    config = deepcopy(_config())
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_simgen_video_only_config(config)
