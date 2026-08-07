import pytest
import yaml

from train_joint_wan_physctrl_trajectory import (
    assert_resume_window_matches,
    configured_future_frames,
)
from training.trajectory_window import TrajectoryWindow


def test_trajectory_trainer_reads_the_configured_pc_horizon():
    assert (
        configured_future_frames(
            {"trajectory": {"start_frame": 5, "future_frames": 24}}
        )
        == 24
    )


def test_trajectory_trainer_rejects_resume_from_another_window(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "trajectory:\n  start_frame: 1\n  future_frames: 48\n"
    )

    with pytest.raises(ValueError, match="trajectory window"):
        assert_resume_window_matches(tmp_path, TrajectoryWindow(5, 24), "latest")


def test_trajectory_training_config_declares_a_fresh_fixed_window():
    with open("configs/train/joint_wan_physctrl_trajectory_832x480.yaml") as handle:
        config = yaml.safe_load(handle)

    assert config["trajectory"] == {"start_frame": 1, "future_frames": 48}
    assert config["objective"]["enable_rigid_loss"] is False
    assert config["objective"]["enable_deform_loss"] is False
    assert config["training"]["resume_from_checkpoint"] is None
