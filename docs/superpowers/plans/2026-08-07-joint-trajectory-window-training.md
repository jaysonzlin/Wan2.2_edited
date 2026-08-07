# Joint Trajectory-Window Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate joint Wan--PhysCtrl training entrypoint that trains a fixed, Wan-compatible window of `N` future point-cloud and video frames starting at source frame `i`.

**Architecture:** Keep the 49-frame dataset and existing trainer intact. Add a small trajectory-window module that validates `(i, N)`, slices both modalities, and derives per-window condition velocities. Generalize the shared PC DDPM batch and joint-model shape checks, then create a dedicated trainer/config that uses the windowed batch, builds an `N`-frame PC model, and excludes rigid/deformation objectives.

**Tech Stack:** Python 3, PyTorch, pytest, h5py, YAML, Accelerate, Wan2.2.

## Global Constraints

- Source samples remain exactly 49 frames with ordered point correspondences.
- `trajectory.future_frames` is an integer multiple of four in `[4, 48]`.
- `trajectory.start_frame` is an integer in `[1, 49 - trajectory.future_frames]`.
- The condition is frame `i - 1`; PC/video targets are frames `i` through `i + N - 1`.
- At `i=1`, derive velocity with a forward per-frame difference; otherwise use a centered per-frame difference at `i-1`.
- Linear velocity is centroid velocity; angular velocity is the least-squares rigid angular component after removing translation.
- The new trajectory trainer starts fresh by default and rejects resume checkpoints created for another `(i, N)` window.
- Rigid and deformation objectives remain disabled in the trajectory trainer.

---

## File Structure

- `training/trajectory_window.py`: owns the window contract, tensor slicing, and later-frame velocity derivation.
- `training/trajectory_config.py`: validates the trajectory mapping and the no-auxiliary objective policy after normal joint-config validation.
- `training/joint_objectives.py`: makes the shared multi-object DDPM helper frame-count agnostic.
- `wan/modules/joint_wan_physctrl.py`: replaces fixed 48/49 error contracts with dynamic trajectory-length validation.
- `train_joint_wan_physctrl_trajectory.py`: dedicated joint training entrypoint; it loads full source samples, applies a fixed window, and trains the matching PC horizon.
- `configs/train/joint_wan_physctrl_trajectory_832x480.yaml`: explicit, fresh-run default experiment configuration.
- `tests/test_trajectory_window.py`: window and kinematics unit tests.
- `tests/test_joint_objectives.py`, `tests/test_joint_bridge.py`, and `tests/test_train_joint_wan_physctrl_trajectory.py`: regression and wiring tests.

### Task 1: Window contract and per-window velocity derivation

**Files:**
- Create: `training/trajectory_window.py`
- Create: `tests/test_trajectory_window.py`

**Interfaces:**
- Produces `TrajectoryWindow(start_frame: int, future_frames: int)`.
- Produces `validate_trajectory_window(window: TrajectoryWindow, source_frames: int = 49) -> None`.
- Produces `window_joint_tensors(video, point_clouds, window) -> tuple[Tensor, Tensor, Tensor, Tensor]`, returning windowed video `[B,N+1,C,H,W]`, point clouds `[B,K,N+1,1,P,3]`, linear velocities `[B,K,1,3]`, and angular velocities `[B,K,1,3]`.

- [ ] **Step 1: Write the failing window-slicing tests**

```python
import pytest
import torch

from training.trajectory_window import TrajectoryWindow, validate_trajectory_window, window_joint_tensors


def test_window_uses_preceding_condition_and_n_contiguous_targets():
    video = torch.arange(49, dtype=torch.float32).reshape(1, 49, 1, 1, 1)
    points = torch.arange(49, dtype=torch.float32).reshape(1, 1, 49, 1, 1, 1).expand(-1, -1, -1, -1, -1, 3)

    windowed_video, windowed_points, _, _ = window_joint_tensors(
        video, points, TrajectoryWindow(start_frame=5, future_frames=8)
    )

    assert windowed_video[:, :, 0, 0, 0].tolist() == [[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    assert windowed_points[0, 0, :, 0, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


@pytest.mark.parametrize("window", [TrajectoryWindow(1, 6), TrajectoryWindow(2, 5), TrajectoryWindow(1, 49)])
def test_window_rejects_non_wan_compatible_or_out_of_range_settings(window):
    with pytest.raises(ValueError):
        validate_trajectory_window(window)
```

- [ ] **Step 2: Run the new tests and verify they fail because the module is absent**

Run: `pytest tests/test_trajectory_window.py -v`

Expected: import failure for `training.trajectory_window`.

- [ ] **Step 3: Add the minimal window contract and slicing implementation**

```python
@dataclass(frozen=True)
class TrajectoryWindow:
    start_frame: int
    future_frames: int


def validate_trajectory_window(window: TrajectoryWindow, source_frames: int = 49) -> None:
    if not isinstance(window.start_frame, int) or isinstance(window.start_frame, bool):
        raise ValueError("trajectory.start_frame must be an integer")
    if not isinstance(window.future_frames, int) or isinstance(window.future_frames, bool):
        raise ValueError("trajectory.future_frames must be an integer")
    if not 4 <= window.future_frames <= 48 or window.future_frames % 4:
        raise ValueError("trajectory.future_frames must be a multiple of four in [4, 48]")
    if not 1 <= window.start_frame <= source_frames - window.future_frames:
        raise ValueError("trajectory.start_frame does not leave enough source frames")
```

Slice `video[:, i-1:i+N]` and `point_clouds[:, :, i-1:i+N]`; reject tensor shapes whose time dimension is not `source_frames` before slicing.

- [ ] **Step 4: Add the failing velocity tests**

```python
def test_window_uses_forward_velocity_at_first_target_frame():
    video = torch.zeros((1, 49, 1, 1, 1))
    points = torch.zeros((1, 1, 49, 1, 4, 3))
    points[:, :, 1:, :, :, 0] = torch.arange(1, 49).reshape(1, 1, 48, 1, 1)

    _, _, linear, angular = window_joint_tensors(video, points, TrajectoryWindow(1, 4))

    assert torch.allclose(linear, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    assert torch.allclose(angular, torch.zeros_like(angular), atol=1e-6)


def test_window_uses_centered_rigid_velocity_for_later_condition():
    # Four non-coplanar, corresponded particles rotating around z once per frame.
    base = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
    points = torch.stack([base @ torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).matrix_power(t).T for t in range(49)])
    video = torch.zeros((1, 49, 1, 1, 1))

    _, _, linear, angular = window_joint_tensors(video, points.reshape(1, 1, 49, 1, 4, 3), TrajectoryWindow(3, 4))

    assert torch.allclose(linear, torch.zeros_like(linear), atol=1e-6)
    assert angular[0, 0, 0, 2].abs() > 0
```

- [ ] **Step 5: Run the velocity tests and verify they fail**

Run: `pytest tests/test_trajectory_window.py -k velocity -v`

Expected: assertions fail because returned velocities are not implemented.

- [ ] **Step 6: Implement per-frame linear and angular estimates**

Compute condition-frame particle velocities from `x[1]-x[0]` at `i=1` or `(x[i]-x[i-2]) / 2` otherwise. Compute centroid mean over points for linear velocity. For each centered particle position `r` and residual velocity `u`, solve the regularized batched least-squares system `u = -[r]_x omega` using `torch.linalg.solve`; use `1e-8 * I` regularization. Return zero angular velocity for degenerate point clouds rather than producing NaNs.

- [ ] **Step 7: Run all window tests**

Run: `pytest tests/test_trajectory_window.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the contract**

```bash
git add training/trajectory_window.py tests/test_trajectory_window.py
git commit -m "feat: add fixed trajectory windows"
```

### Task 2: Validate trajectory configuration and generalize PC DDPM frame counts

**Files:**
- Create: `training/trajectory_config.py`
- Modify: `training/joint_objectives.py:16-48`
- Modify: `tests/test_joint_objectives.py:22-43`
- Modify: `tests/test_trajectory_window.py`

**Interfaces:**
- Produces `load_trajectory_joint_config(path: str | Path, overrides: list[str]) -> dict`.
- Produces `validate_trajectory_joint_config(config: dict) -> None`.
- Keeps `make_aligned_multi_object_pc_ddpm_batch(point_clouds, scheduler, generator)` but accepts any `[B,K,T,1,P,3]` with `T >= 2` and returns `T-1` targets plus `T` frame-time slots.

- [ ] **Step 1: Write failing config and variable-horizon DDPM tests**

```python
from training.trajectory_config import load_trajectory_joint_config


def test_trajectory_config_accepts_a_fixed_wan_compatible_window(tmp_path):
    path = tmp_path / "trajectory.yaml"
    path.write_text(_valid_config() + "\ntrajectory:\n  start_frame: 5\n  future_frames: 24\n")

    config = load_trajectory_joint_config(path, [])

    assert config["trajectory"] == {"start_frame": 5, "future_frames": 24}
    assert config["objective"]["enable_rigid_loss"] is False
    assert config["objective"]["enable_deform_loss"] is False


def test_multi_object_ddpm_preserves_a_24_frame_window_length():
    points = torch.zeros((1, 2, 25, 1, 2, 3))
    batch = make_aligned_multi_object_pc_ddpm_batch(points, RecordingDDPMScheduler(), torch.Generator().manual_seed(1))

    assert batch.model_input.shape == (1, 2, 24, 1, 2, 3)
    assert batch.target.shape == (1, 2, 24, 1, 2, 3)
    assert batch.frame_times.shape == (1, 2, 25)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest tests/test_trajectory_window.py tests/test_joint_objectives.py -k 'trajectory_config or 24_frame_window' -v`

Expected: missing `training.trajectory_config` and a 49-frame validation error.

- [ ] **Step 3: Implement trajectory config validation**

Call `load_joint_config` first, require `trajectory` to be a mapping with only positive integer `start_frame` and `future_frames`, pass a `TrajectoryWindow` to `validate_trajectory_window`, and reject either `objective.enable_rigid_loss` or `objective.enable_deform_loss` when true. Do not mutate the general joint-config validator, so the existing trainer remains valid.

- [ ] **Step 4: Generalize the DDPM helper without changing its 49-frame behavior**

Replace the literal `49` checks and expansions with `frame_count = point_clouds.shape[2]`; require `frame_count >= 2`; build `frame_times` by expanding to `frame_count`. Keep one scheduler timestep per source video and independent Gaussian noise per object.

- [ ] **Step 5: Run focused and existing objective/config tests**

Run: `pytest tests/test_trajectory_window.py tests/test_joint_objectives.py tests/test_joint_dataset.py -q`

Expected: PASS.

- [ ] **Step 6: Commit config and DDPM support**

```bash
git add training/trajectory_config.py training/joint_objectives.py tests/test_trajectory_window.py tests/test_joint_objectives.py
git commit -m "feat: configure trajectory window horizons"
```

### Task 3: Make the joint model’s public validation horizon-agnostic

**Files:**
- Modify: `wan/modules/joint_wan_physctrl.py:201-207`
- Modify: `tests/test_joint_bridge.py`

**Interfaces:**
- `JointWanPhysCtrlModel.forward(...)` accepts point trajectories with any positive temporal length accepted by its `pc_model`, provided `frame_times.shape == [B,K,T+1]`.

- [ ] **Step 1: Write a failing non-48-frame joint forward test**

```python
def test_joint_model_accepts_the_pc_model_configured_horizon():
    wan = WanModel(
        model_type="ti2v", patch_size=(1, 2, 2), text_len=2, in_dim=1, dim=64,
        ffn_dim=128, freq_dim=16, text_dim=4, out_dim=1, num_heads=1, num_layers=8,
    )
    for block in wan.blocks:
        block.self_attn = _ZeroAttention()
        block.cross_attn = _ZeroAttention()
    model = JointWanPhysCtrlModel(
        wan,
        PCTrajectoryModel(
            n_points=2, n_future_frames=24, latent_dim=64, n_layers=8,
            num_heads=1, objective_type="ddpm",
        ),
    )

    _, trajectories = model(
        video_x=[torch.zeros(1, 1, 2, 2)],
        video_t=torch.zeros(1),
        context=[torch.zeros(1, 4)],
        seq_len=1,
        noisy_future_state=torch.randn(1, 1, 24, 1, 2, 3),
        frame_times=torch.ones(1, 1, 25),
        init_pc=torch.randn(1, 1, 1, 2, 3),
        initial_linear_velocity=torch.zeros(1, 1, 1, 3),
        initial_angular_velocity=torch.zeros(1, 1, 1, 3),
    )

    assert trajectories.shape == (1, 1, 24, 1, 2, 3)
```

- [ ] **Step 2: Run it and verify it fails on the fixed 48/49 contract**

Run: `pytest tests/test_joint_bridge.py -k configured_horizon -v`

Expected: `ValueError` referring to the hard-coded 48/49 shape.

- [ ] **Step 3: Replace literal validation with relation checks**

Require `noisy_future_state.ndim == 6`, `frame_times.ndim == 3`, matching batch/object prefixes, and `frame_times.shape[2] == noisy_future_state.shape[2] + 1`. Retain all existing downstream PC-model validation; only remove the accidental fixed-horizon guard.

- [ ] **Step 4: Run the bridge tests**

Run: `pytest tests/test_joint_bridge.py -q`

Expected: PASS.

- [ ] **Step 5: Commit dynamic joint support**

```bash
git add wan/modules/joint_wan_physctrl.py tests/test_joint_bridge.py
git commit -m "feat: allow variable joint PC horizons"
```

### Task 4: Add the dedicated trajectory-window trainer and configuration

**Files:**
- Create: `train_joint_wan_physctrl_trajectory.py`
- Create: `configs/train/joint_wan_physctrl_trajectory_832x480.yaml`
- Create: `tests/test_train_joint_wan_physctrl_trajectory.py`

**Interfaces:**
- `main(config: dict | None = None) -> None` loads `load_trajectory_joint_config` when no config is supplied.
- `configured_future_frames(config: dict) -> int` returns `config["trajectory"]["future_frames"]`.
- `assert_resume_window_matches(output_dir: Path, window: TrajectoryWindow, setting: str | None) -> None` rejects a selected resume state whose saved `config.yaml` has a different trajectory mapping.

- [ ] **Step 1: Write failing trainer-wiring tests**

```python
import pytest

from train_joint_wan_physctrl_trajectory import assert_resume_window_matches, configured_future_frames
from training.trajectory_window import TrajectoryWindow


def test_trajectory_trainer_builds_the_configured_pc_horizon():
    assert configured_future_frames({"trajectory": {"start_frame": 5, "future_frames": 24}}) == 24


def test_trajectory_trainer_rejects_resume_from_another_window(tmp_path):
    (tmp_path / "config.yaml").write_text("trajectory:\n  start_frame: 1\n  future_frames: 48\n")

    with pytest.raises(ValueError, match="trajectory window"):
        assert_resume_window_matches(tmp_path, TrajectoryWindow(5, 24), "latest")
```

- [ ] **Step 2: Run the wiring tests and verify they fail because the trainer is absent**

Run: `pytest tests/test_train_joint_wan_physctrl_trajectory.py -v`

Expected: import failure for `train_joint_wan_physctrl_trajectory`.

- [ ] **Step 3: Implement the dedicated entrypoint by adapting only essential joint-training flow**

Use the existing joint trainer’s model loading, optimizer creation, scheduler, checkpointing, and metrics conventions. Load `JointWanPhysCtrlDataset` with `expected_frames=49` and `load_deformation_fields=False`; after collation, call `window_joint_tensors` before video encoding and PC DDPM construction. Instantiate `PCTrajectoryModel(n_future_frames=window.future_frames)`. Compute only video flow loss and `per_object_pc_x0_mse`; never import/call rigid or deformation objectives. Use the windowed point clouds for visualization ground truth and condition velocities.

Before writing `output_dir/config.yaml`, call `assert_resume_window_matches` when `resume_from_checkpoint` is set. Store the resolved trajectory configuration in that file so a future matching run can resume. Reuse the existing checkpoint loader after this check.

- [ ] **Step 4: Add the dedicated configuration**

Copy the optimizer, model, training, sampling, and logging values from `joint_wan_physctrl_832x480.yaml`. Set:

```yaml
data:
  num_frames: 49
objective:
  enable_rigid_loss: false
  enable_deform_loss: false
trajectory:
  start_frame: 1
  future_frames: 48
training:
  resume_from_checkpoint: null
logging:
  output_dir: outputs/joint_trajectory_i1_n48
```

- [ ] **Step 5: Run trainer wiring tests**

Run: `pytest tests/test_train_joint_wan_physctrl_trajectory.py -v`

Expected: PASS.

- [ ] **Step 6: Run the focused trajectory regression suite**

Run: `pytest tests/test_trajectory_window.py tests/test_joint_objectives.py tests/test_joint_bridge.py tests/test_train_joint_wan_physctrl_trajectory.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the trainer**

```bash
git add train_joint_wan_physctrl_trajectory.py configs/train/joint_wan_physctrl_trajectory_832x480.yaml tests/test_train_joint_wan_physctrl_trajectory.py
git commit -m "feat: train fixed joint trajectory windows"
```

### Task 5: Verify the existing path and new configuration contract

**Files:**
- Modify: `README.md:32-38`
- Test: `tests/test_submit_832x480.py`

**Interfaces:**
- Documents `trajectory.start_frame` and `trajectory.future_frames` as fixed-run controls and names the dedicated training script/config.

- [ ] **Step 1: Write a failing configuration-presence test**

```python
def test_trajectory_training_config_declares_a_fresh_fixed_window():
    config = yaml.safe_load(Path("configs/train/joint_wan_physctrl_trajectory_832x480.yaml").read_text())

    assert config["trajectory"] == {"start_frame": 1, "future_frames": 48}
    assert config["objective"]["enable_rigid_loss"] is False
    assert config["objective"]["enable_deform_loss"] is False
    assert config["training"]["resume_from_checkpoint"] is None
```

- [ ] **Step 2: Run it and verify it fails until the config is present**

Run: `pytest tests/test_submit_832x480.py -k trajectory_training_config -v`

Expected: missing-file or missing-key failure.

- [ ] **Step 3: Document and validate the dedicated path**

Add a concise README command showing `train_joint_wan_physctrl_trajectory.py --config configs/train/joint_wan_physctrl_trajectory_832x480.yaml trajectory.start_frame=5 trajectory.future_frames=24`. State that `N` must be a multiple of four and no auxiliary PC physics losses run in this experiment.

- [ ] **Step 4: Run the complete relevant suite and script syntax check**

Run: `pytest tests/test_joint_dataset.py tests/test_joint_objectives.py tests/test_joint_bridge.py tests/test_train_joint_wan_physctrl.py tests/test_trajectory_window.py tests/test_train_joint_wan_physctrl_trajectory.py tests/test_submit_832x480.py -q && python -m py_compile train_joint_wan_physctrl_trajectory.py`

Expected: PASS and no compilation output.

- [ ] **Step 5: Commit verification/docs**

```bash
git add README.md tests/test_submit_832x480.py
git commit -m "docs: explain trajectory window training"
```

## Plan self-review

- Spec coverage: Tasks 1--4 implement the fixed window, condition velocity, Wan-compatible horizon, dynamic PC/DDPM/model paths, no-auxiliary objective, fresh configuration, and matching-window resume guard. Task 5 documents and protects the public workflow.
- Placeholder scan: no deferred implementation or unspecified error handling remains; all validation errors and boundary behavior are named.
- Type consistency: `TrajectoryWindow` is the single window type passed from configuration to tensor slicing and resume validation; all tensor shapes use `N` consistently.
