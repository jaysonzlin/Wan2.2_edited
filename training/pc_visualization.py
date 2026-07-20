"""Minimal MP4 comparison renderer for point-cloud trajectories."""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def compute_point_colors(pc_data: np.ndarray) -> np.ndarray:
    """Return Viridis RGBA colors from each object's initial-frame height."""
    initial_heights = np.asarray(pc_data)[0, :, :, 2]
    minimum = initial_heights.min(axis=1, keepdims=True)
    height_range = np.ptp(initial_heights, axis=1, keepdims=True)
    normalized = np.full(initial_heights.shape, 0.5, dtype=np.float64)
    np.divide(initial_heights - minimum, height_range, out=normalized, where=height_range > 0)
    return plt.get_cmap("viridis")(normalized)


def compute_trajectory_errors(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame centroid position error and mean per-point error."""
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction and ground_truth must share shape (frames, objects, points, 3)")
    center_error = np.linalg.norm(prediction.mean(axis=2) - ground_truth.mean(axis=2), axis=-1)
    point_error = np.linalg.norm(prediction - ground_truth, axis=-1)
    return center_error.mean(axis=1), point_error.mean(axis=(1, 2))


def _axis_limits(*point_clouds: np.ndarray) -> tuple[tuple[float, float], ...]:
    flat_points = np.concatenate([np.asarray(point_cloud).reshape(-1, 3) for point_cloud in point_clouds])
    minimum = flat_points.min(axis=0)
    maximum = flat_points.max(axis=0)
    midpoint = (minimum + maximum) / 2
    span = max(maximum[0] - minimum[0], maximum[1] - minimum[1], maximum[2] - minimum[2]) + 1.0
    return (
        (midpoint[0] - span / 2, midpoint[0] + span / 2),
        (midpoint[1] - span / 2, midpoint[1] + span / 2),
        (0.0, span),
    )


def _draw_point_cloud(axis, points, colors, frame_index, axis_limits, title: str) -> None:
    x_lim, y_lim, z_lim = axis_limits
    axis.clear()
    axis.set_xlim(x_lim)
    axis.set_ylim(y_lim)
    axis.set_zlim(z_lim)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(title, fontsize=14)
    axis.grid(True)
    for object_index in range(points.shape[1]):
        object_points = points[frame_index, object_index]
        axis.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            c=colors[object_index],
            s=4,
            alpha=0.8,
            edgecolors="none",
            label=f"Object {object_index} (Instance {object_index})",
        )
    axis.legend(loc="upper right")


def save_pointcloud_comparison_mp4(prediction, ground_truth, output_path, fps: int = 12) -> None:
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction and ground_truth must share shape (frames, objects, points, 3)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    position_error, mean_error = compute_trajectory_errors(prediction, ground_truth)
    axis_limits = _axis_limits(prediction, ground_truth)
    prediction_colors = compute_point_colors(prediction)
    ground_truth_colors = compute_point_colors(ground_truth)
    figure = plt.figure(figsize=(20, 10))
    prediction_axis = figure.add_subplot(121, projection="3d")
    ground_truth_axis = figure.add_subplot(122, projection="3d")
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame in range(prediction.shape[0]):
            _draw_point_cloud(prediction_axis, prediction, prediction_colors, frame, axis_limits, "Prediction")
            _draw_point_cloud(ground_truth_axis, ground_truth, ground_truth_colors, frame, axis_limits, "Ground Truth")
            figure.suptitle(
                f"Frame {frame:03d} / {prediction.shape[0] - 1:03d} | "
                f"PE: {position_error[frame]:.4f} | ME: {mean_error[frame]:.4f}",
                fontsize=18,
            )
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(figure)
