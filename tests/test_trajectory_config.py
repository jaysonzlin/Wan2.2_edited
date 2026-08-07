import pytest

from training.trajectory_config import load_trajectory_joint_config


def _config_text() -> str:
    return """
data:
  dataset_root: training_dataset
  num_frames: 49
  num_points: 2048
  train_batch_size: 1
  max_objects_per_sample: 1
model:
  wan_cross_attention_blocks: 8
  physctrl_blocks: 8
  interaction_dim: 512
  interaction_heads: 8
objective:
  pc_type: ddpm_x0
  text_dropout_probability: 0
optimizer:
  video: {lr: 1.0e-5, betas: [0.9, 0.95], eps: 1.0e-8, weight_decay: 0.1}
  bca: {lr: 1.0e-5, betas: [0.9, 0.95], eps: 1.0e-8, weight_decay: 0.1}
  pc: {lr: 1.0e-4, betas: [0.9, 0.999], eps: 1.0e-8, weight_decay: 0.01}
training:
  denoised_latent_mse_every_steps: 50
  checkpoints_total_limit: 2
  resume_from_checkpoint: null
trajectory:
  start_frame: 5
  future_frames: 24
"""


def test_trajectory_config_accepts_a_fixed_wan_compatible_window(tmp_path):
    path = tmp_path / "trajectory.yaml"
    path.write_text(_config_text())

    config = load_trajectory_joint_config(path, [])

    assert config["trajectory"] == {"start_frame": 5, "future_frames": 24}
    assert config["objective"]["enable_rigid_loss"] is False
    assert config["objective"]["enable_deform_loss"] is False


def test_trajectory_config_rejects_enabled_auxiliary_losses(tmp_path):
    path = tmp_path / "trajectory.yaml"
    path.write_text(_config_text().replace("  text_dropout_probability: 0", "  text_dropout_probability: 0\n  enable_deform_loss: true"))

    with pytest.raises(ValueError, match="enable_deform_loss"):
        load_trajectory_joint_config(path, [])
