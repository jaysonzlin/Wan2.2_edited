"""Configuration contract for fixed-window joint trajectory experiments."""

from pathlib import Path

from training.joint_config import load_joint_config
from training.trajectory_window import TrajectoryWindow, validate_trajectory_window


def load_trajectory_joint_config(path: str | Path, overrides: list[str]) -> dict:
    """Load normal joint settings plus one validated, no-auxiliary trajectory window."""
    config = load_joint_config(path, overrides)
    objective = config["objective"]
    objective.setdefault("enable_rigid_loss", False)
    objective.setdefault("enable_deform_loss", False)
    if objective["enable_rigid_loss"]:
        raise ValueError("trajectory training requires objective.enable_rigid_loss=false")
    if objective["enable_deform_loss"]:
        raise ValueError("trajectory training requires objective.enable_deform_loss=false")

    trajectory = config.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be a mapping")
    if set(trajectory) != {"start_frame", "future_frames"}:
        raise ValueError("trajectory must contain only start_frame and future_frames")
    validate_trajectory_window(
        TrajectoryWindow(
            start_frame=trajectory["start_frame"],
            future_frames=trajectory["future_frames"],
        ),
        source_frames=config["data"]["num_frames"],
    )
    return config
