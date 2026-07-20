# PC Visualization Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wan's PC training comparison MP4 visually and numerically equivalent to edited-physctrl's PC comparison visualization.

**Architecture:** Keep `save_pointcloud_comparison_mp4` as Wan's public renderer. Focused helpers calculate height colors, error metrics, shared axes, and one plotted pane; the public writer composes them and owns MP4 cleanup.

**Tech Stack:** Python 3.10, NumPy, Matplotlib 3D, ImageIO/FFmpeg, pytest.

## Global Constraints

- Preserve `save_pointcloud_comparison_mp4(prediction, ground_truth, output_path, fps=12)`, its output path, and its cadence.
- Keep Wan self-contained; do not import `edited-physctrl` or add dependencies.
- Accept only equal `(frames, objects, points, 3)` arrays and reject others with `ValueError`.
- Match PhysCtrl: initial-height Viridis colors per object; shared cubic axes with Z zero; XYZ labels, grid, legends, panes, and PE/ME frame title.
- Do not change the PC training/model/data/flow/scheduler contract.

---

## File structure

- Modify `training/pc_visualization.py`: self-contained comparison helpers and renderer.
- Modify `tests/test_pc_visualization.py`: numerical and MP4 regression coverage.

### Task 1: Add PhysCtrl-equivalent numerical helpers

**Files:**
- Modify: `training/pc_visualization.py`
- Modify: `tests/test_pc_visualization.py`

**Interfaces:**
- Produces `compute_point_colors(pc_data: np.ndarray) -> np.ndarray`, RGBA data shaped `(objects, points, 4)` based on each object's initial Z.
- Produces `compute_trajectory_errors(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, per-frame mean centroid and point errors.
- Task 2 consumes both helpers.

- [ ] **Step 1: Write the failing tests**

```python
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
    np.testing.assert_allclose(colors, colors[:, :1])


def test_trajectory_errors_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="share shape"):
        compute_trajectory_errors(np.zeros((1, 1, 2, 3)), np.zeros((1, 1, 3, 3)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n das python -m pytest tests/test_pc_visualization.py -q`

Expected: collection fails because the two helpers do not yet exist.

- [ ] **Step 3: Implement the minimal helpers**

```python
def compute_point_colors(pc_data: np.ndarray) -> np.ndarray:
    initial_heights = np.asarray(pc_data)[0, :, :, 2]
    minimum = initial_heights.min(axis=1, keepdims=True)
    height_range = np.ptp(initial_heights, axis=1, keepdims=True)
    normalized = np.full(initial_heights.shape, 0.5, dtype=np.float64)
    np.divide(initial_heights - minimum, height_range, out=normalized, where=height_range > 0)
    return plt.get_cmap("viridis")(normalized)


def compute_trajectory_errors(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction and ground_truth must share shape (frames, objects, points, 3)")
    center_error = np.linalg.norm(prediction.mean(axis=2) - ground_truth.mean(axis=2), axis=-1)
    point_error = np.linalg.norm(prediction - ground_truth, axis=-1)
    return center_error.mean(axis=1), point_error.mean(axis=(1, 2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n das python -m pytest tests/test_pc_visualization.py -q`

Expected: the new numerical tests and the existing MP4 test pass.

- [ ] **Step 5: Commit the helper behavior**

```bash
git add training/pc_visualization.py tests/test_pc_visualization.py
git commit -m "feat: add PC visualization metrics"
```

### Task 2: Render the complete PhysCtrl-equivalent comparison MP4

**Files:**
- Modify: `training/pc_visualization.py`
- Modify: `tests/test_pc_visualization.py`

**Interfaces:**
- Consumes the Task 1 color and error helpers.
- Produces unchanged `save_pointcloud_comparison_mp4` with PhysCtrl-equivalent panels and annotations.

- [ ] **Step 1: Write the failing renderer-contract tests**

```python
from training.pc_visualization import _axis_limits


def test_axis_limits_are_cubic_and_start_z_at_zero():
    points = np.array([[[[2.0, -3.0, 4.0], [6.0, 1.0, 8.0]]]], dtype=np.float32)

    x_lim, y_lim, z_lim = _axis_limits(points, points)

    assert x_lim[1] - x_lim[0] == y_lim[1] - y_lim[0] == z_lim[1] - z_lim[0]
    assert z_lim[0] == 0.0


def test_comparison_visualization_rejects_invalid_trajectory_shape(tmp_path):
    invalid = np.zeros((2, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="share shape"):
        save_pointcloud_comparison_mp4(invalid, invalid, tmp_path / "comparison.mp4")
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `conda run -n das python -m pytest tests/test_pc_visualization.py::test_axis_limits_are_cubic_and_start_z_at_zero -q`

Expected: collection fails because `_axis_limits` does not yet exist.

- [ ] **Step 3: Add PhysCtrl-equivalent drawing helpers and compose them in the public writer**

```python
def _axis_limits(*point_clouds: np.ndarray):
    flat = np.concatenate([np.asarray(cloud).reshape(-1, 3) for cloud in point_clouds])
    minimum, maximum = flat.min(axis=0), flat.max(axis=0)
    midpoint = (minimum + maximum) / 2
    span = max(maximum[0] - minimum[0], maximum[1] - minimum[1], maximum[2] - minimum[2]) + 1.0
    return ((midpoint[0] - span / 2, midpoint[0] + span / 2), (midpoint[1] - span / 2, midpoint[1] + span / 2), (0.0, span))


def _draw_point_cloud(axis, points, colors, frame_index, limits, title: str) -> None:
    x_lim, y_lim, z_lim = limits
    axis.clear()
    axis.set_xlim(x_lim); axis.set_ylim(y_lim); axis.set_zlim(z_lim)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X"); axis.set_ylabel("Y"); axis.set_zlabel("Z")
    axis.set_title(title, fontsize=14); axis.grid(True)
    for object_index in range(points.shape[1]):
        cloud = points[frame_index, object_index]
        axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors[object_index], s=4, alpha=0.8, edgecolors="none", label=f"Object {object_index} (Instance {object_index})")
    axis.legend(loc="upper right")
```

Update `save_pointcloud_comparison_mp4` to validate shapes, calculate shared limits/colors/errors once, create `figsize=(20, 10)` left/right 3D axes, call `_draw_point_cloud` with `Prediction` and `Ground Truth`, and set:

```python
figure.suptitle(
    f"Frame {frame_index:03d} / {num_frames - 1:03d} | "
    f"PE: {position_error[frame_index]:.4f} | ME: {mean_error[frame_index]:.4f}",
    fontsize=18,
)
```

Append the RGB canvas each frame and close both writer and figure in `finally`.

- [ ] **Step 4: Run visualization tests to verify they pass**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n das python -m pytest tests/test_pc_visualization.py -q`

Expected: all visualization tests pass and the MP4 test writes a non-empty file.

- [ ] **Step 5: Run the full regression suite**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n das python -m pytest -q`

Expected: all tests pass; only existing Matplotlib/Torch deprecation warnings may remain.

- [ ] **Step 6: Commit the parity renderer**

```bash
git add training/pc_visualization.py tests/test_pc_visualization.py
git commit -m "feat: match PhysCtrl PC visualization"
```
