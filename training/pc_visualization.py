"""Minimal MP4 comparison renderer for point-cloud trajectories."""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def save_pointcloud_comparison_mp4(prediction, ground_truth, output_path, fps: int = 12) -> None:
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction and ground_truth must share shape (frames, objects, points, 3)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate((prediction.reshape(-1, 3), ground_truth.reshape(-1, 3)))
    center = (combined.min(0) + combined.max(0)) / 2
    span = max(float(np.ptp(combined[:, axis])) for axis in range(3)) + 1.0
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, prediction.shape[2]))
    figure = plt.figure(figsize=(12, 6))
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame in range(prediction.shape[0]):
            for index, (title, cloud) in enumerate((("Prediction", prediction), ("Ground truth", ground_truth)), 1):
                axis = figure.add_subplot(1, 2, index, projection="3d")
                points = cloud[frame, 0]
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=4)
                axis.set_title(title)
                axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
                axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
                axis.set_zlim(center[2] - span / 2, center[2] + span / 2)
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
            figure.clear()
    finally:
        writer.close()
        plt.close(figure)
