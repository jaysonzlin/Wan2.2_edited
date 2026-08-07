import pytest
import torch

from training.trajectory_window import (
    TrajectoryWindow,
    validate_trajectory_window,
    window_joint_tensors,
)


def _window_inputs(points: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.arange(49, dtype=torch.float32).reshape(1, 49, 1, 1, 1)
    if points is None:
        points = (
            torch.arange(49, dtype=torch.float32)
            .reshape(1, 1, 49, 1, 1, 1)
            .expand(-1, -1, -1, -1, -1, 3)
        )
    return video, points


def test_window_uses_preceding_condition_and_contiguous_targets():
    video, points = _window_inputs()

    windowed_video, windowed_points, _, _ = window_joint_tensors(
        video, points, TrajectoryWindow(start_frame=5, future_frames=8)
    )

    assert windowed_video[:, :, 0, 0, 0].tolist() == [
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    ]
    assert windowed_points[0, 0, :, 0, 0, 0].tolist() == [
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]


@pytest.mark.parametrize(
    "window",
    [TrajectoryWindow(1, 6), TrajectoryWindow(2, 5), TrajectoryWindow(1, 49)],
)
def test_window_rejects_non_wan_compatible_or_out_of_range_settings(window):
    with pytest.raises(ValueError):
        validate_trajectory_window(window)


def test_window_uses_forward_velocity_at_first_target_frame():
    points = torch.zeros((1, 1, 49, 1, 4, 3))
    points[:, :, 1:, :, :, 0] = torch.arange(1, 49).reshape(1, 1, 48, 1, 1)
    video, points = _window_inputs(points)

    _, _, linear, angular = window_joint_tensors(
        video, points, TrajectoryWindow(1, 4)
    )

    assert torch.allclose(linear, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    assert torch.allclose(angular, torch.zeros_like(angular), atol=1e-6)


def test_window_uses_centered_rigid_velocity_for_later_condition():
    base = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    omega = torch.tensor([0.0, 0.0, 0.25])
    velocity = torch.cross(omega.expand_as(base), base, dim=-1)
    points = base.expand(49, -1, -1).clone()
    points[1] = base - velocity
    points[3] = base + velocity
    points = points.reshape(1, 1, 49, 1, 4, 3)
    video, points = _window_inputs(points)

    _, _, linear, angular = window_joint_tensors(
        video, points, TrajectoryWindow(3, 4)
    )

    assert torch.allclose(linear, torch.zeros_like(linear), atol=1e-6)
    assert torch.allclose(angular, omega.reshape(1, 1, 1, 3), atol=1e-6)
