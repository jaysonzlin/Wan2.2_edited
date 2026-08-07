"""Fixed temporal windows and condition velocities for joint trajectory training."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryWindow:
    """A source-frame index and the number of contiguous future frames to predict."""

    start_frame: int
    future_frames: int


def validate_trajectory_window(
    window: TrajectoryWindow, source_frames: int = 49
) -> None:
    """Reject windows that do not fit the fixed source clip or Wan temporal format."""
    if not isinstance(window.start_frame, int) or isinstance(window.start_frame, bool):
        raise ValueError("trajectory.start_frame must be an integer")
    if not isinstance(window.future_frames, int) or isinstance(
        window.future_frames, bool
    ):
        raise ValueError("trajectory.future_frames must be an integer")
    if not 4 <= window.future_frames <= 48 or window.future_frames % 4:
        raise ValueError(
            "trajectory.future_frames must be a multiple of four in [4, 48]"
        )
    if not 1 <= window.start_frame <= source_frames - window.future_frames:
        raise ValueError("trajectory.start_frame does not leave enough source frames")


def _condition_velocities(
    point_clouds: torch.Tensor, window: TrajectoryWindow
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate centroid and rigid angular velocities at the window condition frame."""
    positions = point_clouds.squeeze(3)
    condition_index = window.start_frame - 1
    if window.start_frame == 1:
        particle_velocity = positions[:, :, 1] - positions[:, :, 0]
    else:
        particle_velocity = (
            positions[:, :, condition_index + 1]
            - positions[:, :, condition_index - 1]
        ) / 2.0

    condition = positions[:, :, condition_index]
    linear = particle_velocity.mean(dim=2)
    centered_positions = condition - condition.mean(dim=2, keepdim=True)
    centered_velocity = particle_velocity - linear.unsqueeze(2)
    x, y, z = centered_positions.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    cross_matrix = torch.stack(
        (
            torch.stack((zeros, z, -y), dim=-1),
            torch.stack((-z, zeros, x), dim=-1),
            torch.stack((y, -x, zeros), dim=-1),
        ),
        dim=-2,
    )
    normal = cross_matrix.transpose(-1, -2).matmul(cross_matrix).sum(dim=2)
    rhs = (
        cross_matrix.transpose(-1, -2)
        .matmul(centered_velocity.unsqueeze(-1))
        .sum(dim=2)
    )
    identity = torch.eye(3, dtype=normal.dtype, device=normal.device)
    angular = torch.linalg.solve(normal + identity * 1e-8, rhs).squeeze(-1)
    return linear.unsqueeze(2), angular.unsqueeze(2)


def window_joint_tensors(
    video: torch.Tensor, point_clouds: torch.Tensor, window: TrajectoryWindow
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice aligned video/point-cloud windows and derive their condition velocities."""
    if video.ndim != 5:
        raise ValueError("video must have shape [B, T, C, H, W]")
    if point_clouds.ndim != 6 or point_clouds.shape[3] != 1 or point_clouds.shape[-1] != 3:
        raise ValueError("point_clouds must have shape [B, K, T, 1, N, 3]")
    if video.shape[0] != point_clouds.shape[0] or video.shape[1] != point_clouds.shape[2]:
        raise ValueError("video and point_clouds must have aligned batch and frame dimensions")
    validate_trajectory_window(window, source_frames=video.shape[1])
    start = window.start_frame - 1
    stop = window.start_frame + window.future_frames
    linear, angular = _condition_velocities(point_clouds, window)
    return (
        video[:, start:stop],
        point_clouds[:, :, start:stop],
        linear,
        angular,
    )
