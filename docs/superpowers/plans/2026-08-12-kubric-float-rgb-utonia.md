# Kubric Float RGB Utonia Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the Utonia PC overfit trainer to consume Kubric object RGB stored as finite `float32 [2048, 3]` values in `[0, 1]` without altering HDF5 inputs.

**Architecture:** Centralize RGB validation and conversion in `training.utonia_features`: validation preserves native array dtype/value for cache fingerprints, while a separate extractor-bound helper produces float32 `[0,255]` immediately before Utonia's `NormalizeColor`. Dataset construction delegates validation to that shared contract, avoiding split rules between cache and data loading.

**Tech Stack:** Python, NumPy, h5py, PyTorch, pytest, Utonia PTv3 preprocessing.

## Global Constraints

- Accept only `uint8 [2048, 3]` values in `[0,255]` or floating `[2048, 3]` values in `[0,1]`; reject non-finite values, other dtypes, shapes, and ranges.
- Do not modify source `pc.hdf5` files.
- Cache fingerprints must use stored RGB values and dtype before conversion.
- Utonia receives float32 color values in `[0,255]` so its existing `NormalizeColor` produces the same `[0,1]` scale as documented.
- Leave coordinate-only training and all unrelated dirty files unchanged.

---

## File map

| File | Responsibility |
| --- | --- |
| `training/utonia_features.py` | Define one RGB contract and scale only at the real Utonia extractor boundary. |
| `training/pc_dataset.py` | Reuse shared RGB validation for selected-object input discovery. |
| `tests/test_utonia_features.py` | Test RGB acceptance/rejection and pre-Utonia scaling without CUDA. |
| `tests/test_pc_dataset.py` | Test Kubric float-RGB object discovery while retaining uint8 coverage. |

## Tasks

### Task 1: Centralize stored-RGB validation and Utonia color preparation

**Files:**
- Modify: `training/utonia_features.py`
- Modify: `tests/test_utonia_features.py`

**Interfaces:**
- Produces: `validate_utonia_rgb(rgb: np.ndarray, *, point_count: int = 2048) -> None`
- Produces: `prepare_utonia_color(rgb: np.ndarray) -> np.ndarray`, returning finite `float32 [N,3]` in `[0,255]`.
- Consumes: `_read_source(path)` and `UtoniaFeatureExtractor.__call__` existing raw RGB path.

- [ ] **Step 1: Write failing contract tests.** Add a parameterized test accepting `np.uint8` color values at 0 and 255 plus `np.float32` unit-range colors at 0 and 1. Assert `prepare_utonia_color` produces `float32` and values 0 and 255 respectively. Add rejection cases for `float32` values below 0/above 1, NaN, wrong shape, and an unsupported integer dtype.

```python
def test_prepare_utonia_color_scales_kubric_unit_float_rgb():
    rgb = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    prepared = prepare_utonia_color(rgb)

    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, [[0.0, 127.5, 255.0]])
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_utonia_features.py`

Expected: collection fails because `prepare_utonia_color` does not exist, or assertions fail because the current code rejects float RGB.

- [ ] **Step 3: Implement the shared contract.** Add a validation helper that checks exact `(N,3)` shape, permits `np.uint8` range `[0,255]` and floating dtypes with finite range `[0,1]`, and raises targeted `ValueError`s. Add a preparation helper that calls validation, converts `uint8` directly to float32, and multiplies accepted floating arrays by `255.0`. Change `_read_source` to call validation but return the untouched stored array. Change `UtoniaFeatureExtractor.__call__` to use `prepare_utonia_color(rgb)` only for its `point["color"]`; keep its seed/fingerprint based on raw RGB.

```python
def prepare_utonia_color(rgb: np.ndarray) -> np.ndarray:
    validate_utonia_rgb(rgb, point_count=rgb.shape[0])
    color = np.asarray(rgb, dtype=np.float32)
    return color if rgb.dtype == np.uint8 else color * 255.0
```

- [ ] **Step 4: Run focused tests to verify green.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_utonia_features.py`

Expected: all feature-cache tests pass, including float RGB conversion and invalid input cases, with no CUDA/Utonia import.

- [ ] **Step 5: Commit.**

```bash
git add training/utonia_features.py tests/test_utonia_features.py
git commit -m "feat: accept Kubric float RGB for Utonia"
```

### Task 2: Reuse the contract for object-dataset validation

**Files:**
- Modify: `training/pc_dataset.py`
- Modify: `tests/test_pc_dataset.py`

**Interfaces:**
- Consumes: `validate_utonia_rgb(rgb, point_count=self.expected_points)` from `training.utonia_features`.
- Produces: object-mode `PCTrajectoryDataset` acceptance for Kubric `float32 [2048,3]` RGB in `[0,1]`.

- [ ] **Step 1: Write a failing Kubric fixture test.** Create `sample_0/objects/000/pc.hdf5` with float32 RGB values including `0.0`, `0.5`, and `1.0`; assert `PCTrajectoryDataset(tmp_path, object_id="000")` constructs and returns its source point tensor. Retain the existing invalid-RGB test cases and add a float value above 1.0.

```python
def test_object_dataset_accepts_kubric_unit_float_rgb(tmp_path):
    rgb = np.full((2048, 3), 0.5, dtype=np.float32)
    write_pc_sample(tmp_path / "sample_0" / "objects" / "000", rgb=rgb)

    assert PCTrajectoryDataset(tmp_path, object_id="000")[0]["points_src"].shape == (1, 2048, 3)
```

- [ ] **Step 2: Run the targeted test to verify it fails.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_pc_dataset.py::test_object_dataset_accepts_kubric_unit_float_rgb`

Expected: fails with the current `rgb must have dtype uint8` validation.

- [ ] **Step 3: Replace duplicated validation.** Import the shared helper and replace the dataset's local dtype/range block with `validate_utonia_rgb(np.asarray(rgb[:]), point_count=self.expected_points)`. Preserve the missing-dataset and shape diagnostics. Do not load or return RGB from `__getitem__`.

- [ ] **Step 4: Run the focused PC suite to verify green.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_pc_dataset.py tests/test_utonia_features.py tests/test_pc_config.py`

Expected: baseline paths, uint8 sources, Kubric unit floats, cache validation, and invalid RGB rejection all pass.

- [ ] **Step 5: Commit.**

```bash
git add training/pc_dataset.py tests/test_pc_dataset.py
git commit -m "feat: validate Kubric RGB in PC dataset"
```

### Task 3: Verify the complete repository state

**Files:**
- Verify only; do not alter unrelated dirty files.

- [ ] **Step 1: Run the full suite.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q`

Expected: all repository tests pass; CPU-only CUDA-autocast warnings may remain.

- [ ] **Step 2: Check the final worktree.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only the RGB-support commits and the user's pre-existing dirty files are present.
