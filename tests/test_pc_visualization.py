import numpy as np
import pytest

from training.pc_visualization import (
    compute_point_colors,
    compute_trajectory_errors,
    save_pointcloud_comparison_mp4,
)


def test_trajectory_errors_match_physctrl_metrics():
    ground_truth = np.zeros((2, 1, 2, 3), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[0, 0, :, 0] = 3.0
    prediction[1, 0, 0, 1] = 4.0

    position_error, mean_error = compute_trajectory_errors(prediction, ground_truth)

    np.testing.assert_allclose(position_error, [3.0, 2.0])
    np.testing.assert_allclose(mean_error, [3.0, 2.0])


def test_point_colors_are_stable_for_a_flat_object():
    colors = compute_point_colors(np.zeros((2, 1, 3, 3), dtype=np.float32))

    assert colors.shape == (1, 3, 4)
    np.testing.assert_allclose(colors, np.broadcast_to(colors[:, :1], colors.shape))


def test_trajectory_errors_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="share shape"):
        compute_trajectory_errors(np.zeros((1, 1, 2, 3)), np.zeros((1, 1, 3, 3)))


def test_comparison_visualization_writes_mp4(tmp_path):
    trajectory = np.zeros((2, 1, 2, 3), dtype=np.float32)
    output = tmp_path / "comparison.mp4"

    save_pointcloud_comparison_mp4(trajectory, trajectory, output, fps=1)

    assert output.is_file()
    assert output.stat().st_size > 0
