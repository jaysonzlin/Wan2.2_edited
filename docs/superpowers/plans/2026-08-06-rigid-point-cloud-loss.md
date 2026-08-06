# Rigid Point-Cloud Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable, per-object rigid point-cloud geometry loss to joint Wan--PhysCtrl training and log it at every optimizer step.

**Architecture:** A pure objective function computes each object’s normalized k-nearest-neighbor edge-length drift from the fixed input point cloud. The joint trainer combines its sum with the existing video and x0 losses only when the configured weight is positive, while logging the aggregate and each object’s scalar on every step.

**Tech Stack:** Python 3, PyTorch, PyTest, Hugging Face Accelerate, YAML.

## Global Constraints

- Rigid objects only; do not add MPM datasets, `vol`, `F`, `C`, or an MPM loss.
- Use fixed frame-zero point clouds to construct the k-NN graph.
- Use scale-normalized squared edge-length drift and `epsilon=1e-8`.
- Default `objective.rigid_loss_weight` to `0.0` and default `objective.rigid_loss_neighbors` to `16` when omitted from legacy configs.
- Log rigid diagnostics even when the configured weight is zero.
- Preserve the joint trainer’s batch-size-one contract and its unaveraged sum across objects.

---

## File Structure

- `training/joint_objectives.py` owns the differentiable per-object geometric objective.
- `training/joint_config.py` validates optional rigid-loss configuration values.
- `train_joint_wan_physctrl.py` composes the loss and converts per-object scalars to tracker metrics.
- `configs/train/joint_wan_physctrl_832x480.yaml` exposes the opted-out defaults.
- `tests/test_joint_objectives.py`, `tests/test_joint_dataset.py`, and `tests/test_train_joint_wan_physctrl.py` cover the three boundaries above.

### Task 1: Per-object rigid geometry objective

**Files:**
- Modify: `training/joint_objectives.py`
- Test: `tests/test_joint_objectives.py`

**Interfaces:**
- Consumes: initial clouds `[B, K, 1, N, 3]`, predictions `[B, K, T, 1, N, 3]`, integer `neighbors`.
- Produces: `per_object_rigid_edge_length_loss(initial_point_clouds, prediction, neighbors, epsilon=1e-8) -> torch.Tensor` with shape `[B, K]`.

- [ ] **Step 1: Write failing invariance and deformation tests**

```python
def test_per_object_rigid_edge_length_loss_is_zero_for_rigid_motion():
    initial = _initial_clouds_for_two_objects()
    prediction = _rigidly_transform_each_object(initial, frames=48)

    losses = per_object_rigid_edge_length_loss(initial, prediction, neighbors=2)

    assert losses.shape == (1, 2)
    assert torch.allclose(losses, torch.zeros_like(losses), atol=1e-6)


def test_per_object_rigid_edge_length_loss_is_positive_for_nonrigid_motion():
    initial = _initial_clouds_for_two_objects()
    prediction = initial[:, :, None].expand(-1, -1, 48, -1, -1, -1).clone()
    prediction[:, 1, :, :, 0] *= 2

    losses = per_object_rigid_edge_length_loss(initial, prediction, neighbors=2)

    assert losses[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert losses[0, 1].item() > 0
```

Use a non-collinear four-point cloud so a rotation and translation are unambiguously rigid. Import `pytest` and the new function at the top of the test module.

- [ ] **Step 2: Run the new objective tests and verify they fail because the function is absent**

Run: `pytest tests/test_joint_objectives.py -k rigid_edge_length -v`

Expected: import or attribute failure naming `per_object_rigid_edge_length_loss`.

- [ ] **Step 3: Implement the minimal objective**

```python
def per_object_rigid_edge_length_loss(
    initial_point_clouds: torch.Tensor,
    prediction: torch.Tensor,
    neighbors: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if initial_point_clouds.ndim != 5 or initial_point_clouds.shape[2] != 1:
        raise ValueError("initial_point_clouds must have shape [B, K, 1, N, 3]")
    if prediction.ndim != 6 or prediction.shape[3] != 1:
        raise ValueError("prediction must have shape [B, K, T, 1, N, 3]")
    if prediction.shape[:2] != initial_point_clouds.shape[:2] or prediction.shape[-2:] != initial_point_clouds.shape[-2:]:
        raise ValueError("initial_point_clouds and prediction must agree on batch, object, point, and coordinate dimensions")
    point_count = initial_point_clouds.shape[-2]
    if not isinstance(neighbors, int) or isinstance(neighbors, bool) or not 1 <= neighbors < point_count:
        raise ValueError("neighbors must be an integer in [1, point_count)")
```

After validation, remove the singleton dimensions. Use `torch.cdist` on frame-zero points, select the smallest `neighbors + 1` distances, discard each point’s self-neighbor, and gather only selected neighbor positions (never materialize an `[N, N]` tensor for every future frame). Compute the formula from the spec and return `edge_drift.mean(dim=(2, 3, 4))`.

- [ ] **Step 4: Run the rigid objective tests and verify they pass**

Run: `pytest tests/test_joint_objectives.py -k rigid_edge_length -v`

Expected: both tests PASS.

- [ ] **Step 5: Add failing input-validation coverage**

```python
def test_per_object_rigid_edge_length_loss_rejects_invalid_neighbor_count():
    initial = torch.zeros((1, 1, 1, 4, 3))
    prediction = torch.zeros((1, 1, 48, 1, 4, 3))

    with pytest.raises(ValueError, match="neighbors"):
        per_object_rigid_edge_length_loss(initial, prediction, neighbors=4)
```

- [ ] **Step 6: Run the validation test and verify it passes**

Run: `pytest tests/test_joint_objectives.py -k invalid_neighbor_count -v`

Expected: PASS.

- [ ] **Step 7: Commit the objective slice**

```bash
git add training/joint_objectives.py tests/test_joint_objectives.py
git commit -m "feat: add per-object rigid point-cloud loss"
```

### Task 2: Rigid-loss configuration contract

**Files:**
- Modify: `training/joint_config.py`
- Modify: `configs/train/joint_wan_physctrl_832x480.yaml`
- Test: `tests/test_joint_dataset.py`

**Interfaces:**
- Consumes: `objective.rigid_loss_weight` and `objective.rigid_loss_neighbors` when provided.
- Produces: a validated configuration that accepts absent keys for legacy test/config compatibility and rejects invalid supplied values.

- [ ] **Step 1: Write failing config tests**

```python
def test_joint_config_accepts_rigid_loss_settings(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(_valid_config() + "\nobjective:\n  rigid_loss_weight: 0.25\n  rigid_loss_neighbors: 8\n")

    objective = load_joint_config(path, [])["objective"]

    assert objective["rigid_loss_weight"] == 0.25
    assert objective["rigid_loss_neighbors"] == 8
```

Refactor `_valid_config()` first so it can accept an `objective_extra` string without emitting two YAML `objective` mappings. Add parametrized invalid cases for a negative or boolean weight and for `0`, `2048`, or boolean neighbors.

- [ ] **Step 2: Run the config tests and verify they fail for missing validation**

Run: `pytest tests/test_joint_dataset.py -k rigid_loss -v`

Expected: invalid configurations are accepted, or valid configured values are not available as expected.

- [ ] **Step 3: Implement optional validation and expose defaults**

```python
weight = objective.get("rigid_loss_weight", 0.0)
if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
    raise ValueError("objective.rigid_loss_weight must be a non-negative number")

neighbors = objective.get("rigid_loss_neighbors", 16)
if not isinstance(neighbors, int) or isinstance(neighbors, bool) or not 1 <= neighbors < data["num_points"]:
    raise ValueError("objective.rigid_loss_neighbors must be an integer in [1, data.num_points)")
```

Add `rigid_loss_weight: 0.0` and `rigid_loss_neighbors: 16` under the shipped joint config’s `objective` mapping. Do not mutate loaded legacy configs; the trainer will use the same `.get` defaults.

- [ ] **Step 4: Run the config tests and verify they pass**

Run: `pytest tests/test_joint_dataset.py -k 'joint_config and rigid_loss' -v`

Expected: configured valid values pass; all invalid values raise the specified errors.

- [ ] **Step 5: Commit the configuration slice**

```bash
git add training/joint_config.py configs/train/joint_wan_physctrl_832x480.yaml tests/test_joint_dataset.py
git commit -m "feat: configure rigid point-cloud loss"
```

### Task 3: Joint loss composition and tracker metrics

**Files:**
- Modify: `train_joint_wan_physctrl.py`
- Test: `tests/test_train_joint_wan_physctrl.py`

**Interfaces:**
- Consumes: `per_object_rigid_edge_length_loss(...) -> [B, K]`, `objective.rigid_loss_weight`, and `objective.rigid_loss_neighbors`.
- Produces: total joint loss including the weighted rigid sum and metrics named `train/rigid_loss_sum` plus `train/rigid_loss_object_000`, etc.

- [ ] **Step 1: Write failing composition and metric-name tests**

```python
def test_joint_loss_adds_weighted_rigid_sum_without_object_averaging():
    total = combine_joint_losses(
        video_loss=torch.tensor(2.0),
        object_losses=torch.tensor([[3.0, 5.0]]),
        rigid_loss_sum=torch.tensor(7.0),
        rigid_loss_weight=0.25,
    )

    assert total.item() == 11.75


def test_per_object_metric_values_uses_zero_padded_slots():
    metrics = per_object_metric_values("train/rigid_loss_object", torch.tensor([1.5, 2.5]))

    assert metrics == {
        "train/rigid_loss_object_000": 1.5,
        "train/rigid_loss_object_001": 2.5,
    }
```

- [ ] **Step 2: Run the trainer unit tests and verify they fail for the new API**

Run: `pytest tests/test_train_joint_wan_physctrl.py -k 'weighted_rigid or per_object_metric' -v`

Expected: import/signature failure for the new weighted arguments or metric helper.

- [ ] **Step 3: Implement the minimal trainer integration**

```python
def combine_joint_losses(video_loss, object_losses, rigid_loss_sum, rigid_loss_weight):
    return video_loss + object_losses.sum() + rigid_loss_weight * rigid_loss_sum


def per_object_metric_values(prefix, losses):
    return {f"{prefix}_{index:03d}": value.item() for index, value in enumerate(losses)}
```

Import the rigid objective inside `main()`. After computing x0 losses, compute `rigid_losses` from `point_clouds[:, :, 0]` and `pc_prediction`, sum it without averaging, and call `combine_joint_losses` with the `.get` configuration defaults. Add its aggregate and `per_object_metric_values("train/rigid_loss_object", rigid_losses[0].detach())` to the existing Accelerate metrics mapping. Replace the duplicated PC per-object comprehension with the same helper.

- [ ] **Step 4: Run the trainer unit tests and verify they pass**

Run: `pytest tests/test_train_joint_wan_physctrl.py -v`

Expected: all tests PASS, including the existing unweighted-loss behavior updated to pass `rigid_loss_sum=torch.tensor(0.0)` and `rigid_loss_weight=0.0`.

- [ ] **Step 5: Run the focused regression suite**

Run: `pytest tests/test_joint_objectives.py tests/test_joint_dataset.py tests/test_train_joint_wan_physctrl.py -v`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the trainer slice**

```bash
git add train_joint_wan_physctrl.py tests/test_train_joint_wan_physctrl.py
git commit -m "feat: train and log rigid point-cloud loss"
```
