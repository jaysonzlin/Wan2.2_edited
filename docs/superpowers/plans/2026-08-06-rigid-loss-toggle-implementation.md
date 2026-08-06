# Rigid Loss Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let joint Wan PhysCtrl training skip rigid-loss computation and all rigid-loss logging through one configuration toggle.

**Architecture:** Add a validated `objective.enable_rigid_loss` setting, defaulting to `false`. A small trainer-local helper gates loss evaluation while returning a zero tensor of the normal per-object shape when disabled; a companion helper conditionally adds rigid metrics. This keeps loss aggregation stable without invoking the rigid objective.

**Tech Stack:** Python, PyTorch, PyYAML, pytest.

## Global Constraints

- `objective.enable_rigid_loss: false` is the backward-compatible configuration default.
- When disabled, `per_object_rigid_edge_length_loss` must not execute and no rigid metric keys may be emitted.
- When enabled, retain the existing rigid-loss math, weighting, and metric names.
- Continue validating `rigid_loss_weight` and `rigid_loss_neighbors` independently of the toggle.

## Task 1: Add and validate the configuration flag

**Files:**
- Modify: `configs/joint_wan_physctrl_832x480.yaml`
- Modify: `training/joint_config.py`
- Modify: `tests/test_joint_dataset.py`

- [ ] Add `objective.enable_rigid_loss: false` next to the existing rigid objective settings in the 832×480 configuration.
- [ ] In `load_joint_config`, read the value with `.get(..., False)` and raise a `ValueError` unless it is a real Python boolean. Preserve the present validation for the weight and neighbor count.
- [ ] Add a valid-config regression test proving a true boolean is accepted and available on the loaded config.
- [ ] Add parametrized invalid-config cases such as integer, string, and null values, asserting the validation error mentions `enable_rigid_loss`.

**Run:**

```bash
conda run -n das python -m pytest tests/test_joint_dataset.py -q
```

**Expected:** The config default is false, true is accepted, and non-booleans fail clearly.

## Task 2: Gate rigid computation and rigid logging in the trainer

**Files:**
- Modify: `train_joint_wan_physctrl.py`
- Modify: `tests/test_train_joint_wan_physctrl.py`

- [ ] Add a focused helper that receives the enabled flag, initial point clouds, prediction, neighbor count, and objective callable. When disabled, return a zero `(batch, object)` tensor on the prediction device/dtype without calling the objective; when enabled, call the existing `per_object_rigid_edge_length_loss` with unchanged arguments.
- [ ] Add a focused helper that conditionally appends `train/rigid_loss_sum` and per-object `train/rigid_loss_object/*` metrics only when the flag is enabled.
- [ ] In the training iteration, obtain `enable_rigid_loss` from `config["objective"]`, invoke the computation helper instead of the unconditional objective call, and use the metric helper rather than unconditionally adding rigid metrics.
- [ ] Keep passing the resulting summed tensor to `combine_joint_losses`; disabled mode therefore contributes zero while preserving its existing function contract.
- [ ] Add unit tests that:
  - prove disabled mode returns appropriately shaped zeros and never invokes a deliberately failing objective callable;
  - prove enabled mode delegates with the current arguments and returns its result;
  - prove rigid metric keys are absent when disabled and present with their existing names when enabled.

**Run:**

```bash
conda run -n das python -m pytest tests/test_train_joint_wan_physctrl.py tests/test_joint_dataset.py tests/test_joint_objectives.py -q
```

**Expected:** Both toggle states are covered, disabled training avoids rigid work and metric emission, and existing objective behavior remains intact.

## Task 3: Validate integration and review the change

**Files:**
- Verify: `configs/joint_wan_physctrl_832x480.yaml`
- Verify: `training/joint_config.py`
- Verify: `train_joint_wan_physctrl.py`
- Verify: `tests/test_joint_dataset.py`
- Verify: `tests/test_train_joint_wan_physctrl.py`

- [ ] Run the targeted test suite from Task 2.
- [ ] Inspect the trainer call site to confirm there is no remaining unconditional call or unconditional rigid metric update.
- [ ] Run `git diff --check` and inspect the scoped diff, excluding pre-existing unrelated working-tree changes.

**Run:**

```bash
git diff --check
git diff -- configs/joint_wan_physctrl_832x480.yaml training/joint_config.py train_joint_wan_physctrl.py tests/test_joint_dataset.py tests/test_train_joint_wan_physctrl.py
```

**Expected:** Targeted tests pass and the scoped diff has no whitespace errors or unrelated edits.

