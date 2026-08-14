# Control Object-Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Utonia-free control training load the configured object-level point-cloud trajectory.

**Architecture:** `build_pc_training_dataset` will use the optional `data.object_id` to select the object-level dataset before branching on Utonia. The Utonia-enabled branch retains its cache preparation; the disabled branch returns the object-level dataset without an extractor or cache.

**Tech Stack:** Python, pytest, h5py-backed `PCTrajectoryDataset`.

## Global Constraints

- A configured object ID always maps to `sample_*/objects/<object_id>/pc.hdf5`.
- Configurations with no object ID retain the `sample_*/pc.hdf5` layout.
- Utonia cache preparation occurs only when `model.utonia_enabled` is true.

---

### Task 1: Preserve object selection for the control run

**Files:**
- Modify: `tests/test_train_pc.py`
- Modify: `train_pc.py:127-147`

**Interfaces:**
- Consumes: `config["data"].get("object_id")`, `config["model"].get("utonia_enabled", False)`.
- Produces: `build_pc_training_dataset(...) -> tuple[dataset, None]` for Utonia-disabled configurations, where `dataset_factory` receives `object_id` when configured.

- [ ] **Step 1: Write the failing test**

```python
def test_build_pc_training_dataset_uses_object_dataset_without_utonia():
    calls = []

    def dataset_factory(root, **kwargs):
        calls.append((root, kwargs))
        return object()

    dataset, feature_dim = build_pc_training_dataset(
        {
            "data": {"dataset_root": "input", "object_id": "000"},
            "model": {"utonia_enabled": False},
        },
        dataset_factory=dataset_factory,
        extractor_factory=lambda _: AssertionError("not called"),
        cache_preparer=lambda *_: AssertionError("not called"),
    )

    assert dataset is not None
    assert feature_dim is None
    assert calls == [("input", {"object_id": "000"})]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_pc.py::test_build_pc_training_dataset_uses_object_dataset_without_utonia -v`

Expected: FAIL because the disabled branch invokes `dataset_factory("input")` without `object_id`.

- [ ] **Step 3: Write minimal implementation**

```python
if not config["model"].get("utonia_enabled", False):
    object_id = data.get("object_id")
    if object_id is None:
        return dataset_factory(data["dataset_root"]), None
    return dataset_factory(data["dataset_root"], object_id=object_id), None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_pc.py::test_build_pc_training_dataset_uses_object_dataset_without_utonia -v`

Expected: PASS.

- [ ] **Step 5: Run focused regression coverage**

Run: `pytest tests/test_train_pc.py tests/test_pc_dataset.py tests/test_submit_control.py -v`

Expected: PASS with the existing top-level and Utonia object-dataset behavior unchanged.
