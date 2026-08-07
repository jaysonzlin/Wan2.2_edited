# Approximate Deformation Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add baseline-corrected, PhysCtrl-style elastic deformation supervision to the TD joint-training sample behind a disabled-by-default configuration toggle.

**Architecture:** A deterministic preprocessing command augments each existing object HDF5 with GT-derived `F`, `C`, uniform particle volume, grid metadata, and its own GT update residual. The joint dataset loads this contract only when requested. A differentiable P2G/G2P objective consumes x0 trajectories and these frozen fields, then adds only residual above the saved GT baseline to the existing joint loss.

**Tech Stack:** Python 3, NumPy, h5py, PyTorch, pytest, YAML.

## Global Constraints

- Do not change point order, image files, or the existing `point_cloud` and velocity datasets.
- Derive every object from its fixed 49-frame, 2,048-point trajectory with a frame-zero 32-neighbor graph.
- Use `dt=0.02`, `grid_size=125`, `grid_lim=10.0`, and uniform volumes.
- Apply the loss per object; never mix particle neighborhoods or grid fields across objects.
- Default `objective.enable_deform_loss` to `false` and `objective.deform_loss_weight` to `0.001`.

---

### Task 1: Define and generate the persistent deformation-field contract

**Files:**
- Create: `Wan2.2_edited/training/derive_deformation_fields.py`
- Modify: `kubric/td_832x480_3_soft/sample_0/objects/000/pc.hdf5`
- Modify: `kubric/td_832x480_3_soft/sample_0/objects/001/pc.hdf5`
- Modify: `kubric/td_832x480_3_soft/sample_0/objects/002/pc.hdf5`
- Test: `Wan2.2_edited/tests/test_deformation_fields.py`

**Interfaces:**
- Consumes: one sample directory containing lexically ordered `objects/*/pc.hdf5` trajectories `(49, 1, 2048, 3)`.
- Produces: `derive_sample_fields(sample_dir: Path, neighbors: int = 32) -> None`, which overwrites only the seven `deform_*` datasets and attributes described in the design.

- [ ] **Step 1: Write the failing contract test**

```python
def test_derive_sample_fields_writes_aligned_deformation_arrays(tmp_path):
    make_td_sample(tmp_path, object_count=2, frames=49, points=8)
    derive_sample_fields(tmp_path, neighbors=3, grid_size=8)
    with h5py.File(tmp_path / "objects/000/pc.hdf5") as source:
        assert source["deform_F"].shape == (49, 1, 8, 3, 3)
        assert source["deform_C"].shape == (49, 1, 8, 3, 3)
        assert source["deform_volume"].shape == (1, 8)
        assert source["deform_baseline"].shape == (47, 1, 8, 3, 3)
        assert source.attrs["deform_dt"] == pytest.approx(0.02)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd Wan2.2_edited && pytest tests/test_deformation_fields.py::test_derive_sample_fields_writes_aligned_deformation_arrays -v`

Expected: FAIL because `training.derive_deformation_fields` does not exist.

- [ ] **Step 3: Implement deterministic MLS field derivation and the HDF5 writer**

```python
def derive_sample_fields(sample_dir: Path, neighbors: int = 32, *, grid_size: int = 125) -> None:
    trajectories = load_all_object_trajectories(sample_dir)
    origin, scale = make_shared_grid_transform(trajectories, grid_size=grid_size)
    for path, trajectory in trajectories:
        grid_positions = world_to_grid(trajectory, origin, scale, grid_size=grid_size)
        F, C = estimate_mls_fields(grid_positions, neighbors=neighbors, dt=0.02)
        baseline = physctrl_update_residual(grid_positions, F, C, np.ones(trajectory.shape[1]))
        write_deform_fields(path, F, C, baseline, origin, scale, neighbors, grid_size)
```

Use frozen KNN indices from frame zero, Gaussian distance weights, ridge-stabilized 3x3 least squares, central velocity differences, and forward/backward endpoint differences. The residual helper must return absolute `(47, N, 3, 3)` entries, not a reduced scalar.

- [ ] **Step 4: Run the contract test and derive the real TD sample fields**

Run: `cd Wan2.2_edited && pytest tests/test_deformation_fields.py::test_derive_sample_fields_writes_aligned_deformation_arrays -v && python -m training.derive_deformation_fields --sample-dir ../kubric/td_832x480_3_soft/sample_0 --neighbors 32`

Expected: PASS; every `objects/000..002/pc.hdf5` contains the new data without changing existing dataset shapes.

- [ ] **Step 5: Verify baseline cancellation on the real fields**

Run: `cd Wan2.2_edited && python -m training.derive_deformation_fields --verify --sample-dir ../kubric/td_832x480_3_soft/sample_0`

Expected: each object reports zero baseline-corrected GT residual and its nonzero raw residual.

### Task 2: Load deformation fields only for enabled training

**Files:**
- Modify: `Wan2.2_edited/training/joint_dataset.py`
- Modify: `Wan2.2_edited/training/joint_config.py`
- Test: `Wan2.2_edited/tests/test_joint_dataset.py`

**Interfaces:**
- Consumes: `load_deformation_fields: bool` passed to `JointWanPhysCtrlDataset`.
- Produces: when enabled, each sample has `deform_F`, `deform_C`, `deform_volume`, `deform_baseline`, `deform_grid_origin`, and `deform_grid_scale` stacked by object.

- [ ] **Step 1: Write failing dataset and configuration tests**

```python
def test_joint_dataset_reads_enabled_deformation_contract(tmp_path):
    sample = make_valid_joint_sample(tmp_path, objects=1)
    add_deform_contract(sample / "objects/000/pc.hdf5")
    item = JointWanPhysCtrlDataset(tmp_path, load_deformation_fields=True)[0]
    assert item["deform_F"].shape == (1, 49, 1, 2048, 3, 3)

def test_joint_config_rejects_non_boolean_deform_toggle(tmp_path):
    with pytest.raises(ValueError, match="enable_deform_loss"):
        load_joint_config(write_joint_config(tmp_path, "enable_deform_loss: 1"), [])
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `cd Wan2.2_edited && pytest tests/test_joint_dataset.py -k 'deformation_contract or deform_toggle' -v`

Expected: FAIL because the loader and validator do not expose deformation fields.

- [ ] **Step 3: Implement the gated loader and validation**

```python
dataset = JointWanPhysCtrlDataset(
    config["data"]["dataset_root"],
    expected_size=(config["data"]["width"], config["data"]["height"]),
    expected_points=config["data"]["num_points"],
    load_deformation_fields=config["objective"].get("enable_deform_loss", False),
)
```

Require every new HDF5 dataset, exact shapes, common grid metadata across objects, and finite values when enabled. Validate `enable_deform_loss` as boolean, `deform_loss_weight` as nonnegative numeric, and `deform_loss_neighbors` as integer in `[1, data.num_points)`.

- [ ] **Step 4: Run the focused tests**

Run: `cd Wan2.2_edited && pytest tests/test_joint_dataset.py -k 'deformation_contract or deform_toggle' -v`

Expected: PASS.

### Task 3: Add the differentiable baseline-corrected deformation objective

**Files:**
- Modify: `Wan2.2_edited/training/joint_objectives.py`
- Test: `Wan2.2_edited/tests/test_joint_objectives.py`

**Interfaces:**
- Produces: `per_object_baseline_corrected_deform_loss(initial_point_clouds, prediction, deform_F, deform_C, deform_volume, deform_baseline, deform_grid_origin, deform_grid_scale, *, dt=0.02, grid_size=125, grid_lim=10.0) -> Tensor` shaped `(B, K)`.

- [ ] **Step 1: Write failing objective tests**

```python
def test_deformation_loss_cancels_saved_gt_baseline():
    loss = per_object_baseline_corrected_deform_loss(
        initial, gt_future, F, C, volume, baseline, origin, scale, grid_size=8
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

def test_deformation_loss_increases_for_perturbed_future():
    clean = per_object_baseline_corrected_deform_loss(...)
    perturbed = per_object_baseline_corrected_deform_loss(..., prediction=gt_future + jitter)
    assert torch.all(perturbed > clean)
```

- [ ] **Step 2: Run the objective tests to verify they fail**

Run: `cd Wan2.2_edited && pytest tests/test_joint_objectives.py -k deform -v`

Expected: FAIL because the objective does not exist.

- [ ] **Step 3: Implement the sequential P2G/G2P loss**

```python
full_prediction = torch.cat((initial_point_clouds.unsqueeze(2), prediction), dim=2)
raw = physctrl_deformation_residual(full_prediction, deform_F, deform_C, deform_volume, origin, scale)
excess = (raw - deform_baseline).clamp_min(0)
return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.01, reduction="none").mean(dim=(2, 3, 4, 5))
```

Implement P2G and G2P one frame at a time to bound grid memory. The helper must preserve gradients through `prediction` and must not require gradients for saved fields. Use the stored shared coordinate transform before grid indexing.

- [ ] **Step 4: Run the focused objective tests**

Run: `cd Wan2.2_edited && pytest tests/test_joint_objectives.py -k deform -v`

Expected: PASS.

### Task 4: Wire, configure, and verify the joint trainer

**Files:**
- Modify: `Wan2.2_edited/train_joint_wan_physctrl.py`
- Modify: `Wan2.2_edited/configs/train/joint_wan_physctrl_832x480.yaml`
- Modify: `Wan2.2_edited/tests/test_train_joint_wan_physctrl.py`

**Interfaces:**
- Consumes: per-object deformation loss `(B, K)` and `objective.enable_deform_loss`.
- Produces: `train/deform_loss_sum` and one `train/deform_loss_object_###` metric when enabled; the weighted loss includes `deform_loss_weight * deform_loss_sum`.

- [ ] **Step 1: Write failing wiring tests**

```python
def test_disabled_deform_loss_returns_zero_without_calling_objective():
    result = deform_loss_terms(False, initial, prediction, fields, deform_loss_fn=raise_if_called)
    assert torch.equal(result, torch.zeros((1, 2)))

def test_joint_total_includes_weighted_deform_loss():
    total = combine_joint_losses(video, pc, rigid, 0.0, deform=torch.tensor(7.0), deform_weight=0.001)
    assert total.item() == pytest.approx(video.item() + pc.sum().item() + 0.007)
```

- [ ] **Step 2: Run the trainer tests to verify they fail**

Run: `cd Wan2.2_edited && pytest tests/test_train_joint_wan_physctrl.py -k deform -v`

Expected: FAIL because the wiring helpers and metrics do not exist.

- [ ] **Step 3: Wire the guarded objective and configuration**

```yaml
enable_deform_loss: false
deform_loss_weight: 0.001
deform_loss_neighbors: 32
```

Compute the loss only when enabled, add its weighted sum to the existing total, and emit deform metrics only when it ran. Keep the existing rigid-loss behavior unchanged.

- [ ] **Step 4: Run all focused regression tests and a loader smoke test**

Run: `cd Wan2.2_edited && pytest tests/test_joint_dataset.py tests/test_joint_objectives.py tests/test_train_joint_wan_physctrl.py -q && python train_joint_wan_physctrl.py --config configs/train/joint_wan_physctrl_832x480.yaml objective.enable_deform_loss=true training.max_train_steps=0`

Expected: PASS; the enabled loader accepts `td_832x480_3_soft`, and the disabled configuration keeps the original execution path.
