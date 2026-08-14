# Utonia History-Conditioned PC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in four-frame Utonia history-conditioned DDPM experiment while preserving the existing velocity-conditioned PC training path and resume support.

**Architecture:** Make conditioning mode an explicit model configuration contract. Split clips into a configurable conditioning prefix and derived future suffix, adapt the model and DDPM helper to compose either velocity or history temporal states, and give the pipeline a history-aware sampling interface. Utonia cache generation and feature fusion remain unchanged.

**Tech Stack:** Python, PyTorch, Accelerate, Diffusers, PyYAML, pytest.

## Global Constraints

- `velocity` remains the default mode and preserves current 1/48 shapes, velocity tokens, model calls, and resume behavior.
- `history` is opt-in, currently requires exactly four history frames, and derives a 45-frame future horizon from the fixed 49-frame clip.
- History-mode DDPM keeps absolute x0 targets and uses timestep zero for known frames and the sampled DDPM timestep for noisy future frames.
- Frame-zero Utonia features remain `[B, N, D]` and are fused into all temporal point tokens.
- Do not modify user-owned dirty files outside this feature's files.

---

### Task 1: Define and test conditioning configuration and dataset splitting

**Files:**
- Modify: `training/pc_config.py`
- Modify: `training/pc_dataset.py`
- Modify: `tests/test_pc_config.py`
- Modify: `tests/test_pc_dataset.py`

**Interfaces:**
- `validate_pc_config(config)` accepts absent/`velocity` conditioning and validates `history` with `history_frames == 4`.
- `PCTrajectoryDataset(..., history_frames: int = 1)` returns `points_src`, `points_tgt`, and, in history mode, `points_history`.

- [ ] Write failing tests for default velocity compatibility, invalid history fields, and the four/45 dataset split.
- [ ] Run those tests and observe failures caused by unsupported history configuration and fields.
- [ ] Add minimal config validation and dataset split implementation without changing the velocity-mode sample contract.
- [ ] Run the focused config/dataset tests until green.

### Task 2: Make objectives and model support both temporal layouts

**Files:**
- Modify: `training/pc_ddpm.py`
- Modify: `training/pc_objectives.py`
- Modify: `wan/modules/pc_trajectory.py`
- Modify: `tests/test_pc_ddpm.py`
- Modify: `tests/test_pc_objectives.py`
- Modify: `tests/test_pc_trajectory_model.py`

**Interfaces:**
- DDPM/flow batch builders accept the number of known frames and return full-sequence frame times.
- `PCTrajectoryModel(..., conditioning="velocity", history_frames=1)` preserves velocity mode; history mode consumes `(B, 4, 1, N, 3)` history and predicts `(B, 45, 1, N, 3)`.

- [ ] Write failing tests for 4/45 DDPM frame times, history model shape, and Utonia expansion across 49 frames.
- [ ] Run focused tests and verify failures reflect the absent history interfaces.
- [ ] Implement the smallest shared temporal-state construction needed for both modes; retain velocity encoders and control tokens only in velocity mode.
- [ ] Run focused objective/model tests until green.

### Task 3: Add history-aware sampling and trainer integration

**Files:**
- Modify: `wan/pc_pipeline.py`
- Modify: `train_pc.py`
- Modify: `tests/test_pc_pipeline.py`
- Modify: `tests/test_train_pc.py`

**Interfaces:**
- History sampler accepts `points_history`, generates the derived future horizon, and returns the future frames for the trainer to prepend during visualization.
- Trainer selects history-aware dataset/model/batch/pipeline calls only when configured, retaining existing velocity control flow otherwise.

- [ ] Write failing pipeline and trainer tests for forwarding four known frames, Utonia, visualization assembly, and preserved velocity calls.
- [ ] Run the focused tests and verify their expected interface failures.
- [ ] Implement conditional wiring in the pipeline and trainer, including existing checkpoint loading for either same-mode run.
- [ ] Run focused pipeline/trainer tests until green.

### Task 4: Create the experiment config and verify integration

**Files:**
- Create: `configs/train/config_pc_utonia_history_overfit.yaml`
- Modify: `tests/test_train_pc.py`

- [ ] Write a failing load test that asserts the new history config's distinct output/run name, null initial resume value, and four-frame history fields.
- [ ] Add the YAML as a like-for-like clone of `config_pc_utonia_overfit.yaml` with only experiment identity and history-conditioning fields changed.
- [ ] Run focused PC tests, `python -m compileall` for modified Python modules, and the best available repository test command.
- [ ] Inspect `git diff --check` and status; commit only the feature files.
