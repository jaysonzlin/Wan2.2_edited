# PC Checkpoint Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cap the Utonia Slurm experiment at 10,000 steps and retain only its two newest Accelerate checkpoints.

**Architecture:** Add a validated top-level checkpoint-retention limit to PC configs and a small numeric-directory pruning helper in `train_pc.py`. Prune only after a successful Accelerate save on the main process. The Utonia launcher supplies the 10,000-step and two-checkpoint overrides while preserving its existing resume behavior.

**Tech Stack:** Python 3.10, Accelerate checkpoints, Bash/Slurm, pytest.

## Global Constraints

- Retain numeric directories named `checkpoint-<step>` only; ignore non-numeric `checkpoint-*` directories.
- Prune after `accelerator.save_state(...)`, only on the main process.
- Preserve the newest `checkpoints_total_limit` numeric directories; require a positive integer limit.
- Keep the base PC configs' `max_train_steps: 60000`; add their default `checkpoints_total_limit: 2`, while the launcher also overrides `max_train_steps=10000` and `checkpoints_total_limit=2`.
- Do not modify Accelerate’s save/load format, checkpoint naming, Utonia feature cache, or latest-resume fallback behavior.

---

## File Map

| File | Responsibility |
| --- | --- |
| `training/pc_config.py` | Validates `checkpoints_total_limit`. |
| `train_pc.py` | Prunes old numeric Accelerate checkpoint directories after saves. |
| `submit_utonia.sh` | Supplies Utonia run-specific max-step and retention overrides. |
| `tests/test_pc_config.py` | Covers retention-limit validation. |
| `tests/test_train_pc.py` | Covers numeric checkpoint pruning. |
| `tests/test_submit_utonia.py` | Pins Utonia launcher overrides. |

## Tasks

### Task 1: Validate the PC checkpoint-retention limit

**Files:**
- Modify: `training/pc_config.py:79-85`
- Modify: `configs/train/config_pc.yaml:21-22`
- Modify: `configs/train/config_pc_utonia_overfit.yaml:21-22`
- Modify: `tests/test_pc_config.py`

**Interfaces:**
- Consumes: top-level config field `checkpoints_total_limit`.
- Produces: `load_pc_config(...)` that requires `checkpoints_total_limit` to be a positive integer.

- [ ] **Step 1: Add the failing validation test.**

```python
@pytest.mark.parametrize("limit", ["0", "-1", "1.5", "null", "invalid"])
def test_pc_config_rejects_invalid_checkpoint_retention_limit(tmp_path, limit):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "lr_scheduler: cosine\n", f"lr_scheduler: cosine\ncheckpoints_total_limit: {limit}\n"
        )
    )
    with pytest.raises(ValueError, match="checkpoints_total_limit must be a positive integer"):
        load_pc_config(path, [])
```

- [ ] **Step 2: Run the validation test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py::test_pc_config_rejects_invalid_checkpoint_retention_limit`

Expected: FAIL because PC config validation does not yet inspect the field.

- [ ] **Step 3: Add the minimal validation.**

```python
limit = config.get("checkpoints_total_limit")
if not isinstance(limit, int) or limit <= 0:
    raise ValueError("checkpoints_total_limit must be a positive integer")
```

- [ ] **Step 4: Add `checkpoints_total_limit: 2` to both `configs/train/config_pc.yaml` and `configs/train/config_pc_utonia_overfit.yaml`; add the same field to `valid_config_text()` in `tests/test_pc_config.py`.** This preserves the validation contract for both direct PC launch configurations and fixture-generated configs.

- [ ] **Step 5: Run PC config tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py`

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add training/pc_config.py configs/train/config_pc.yaml configs/train/config_pc_utonia_overfit.yaml tests/test_pc_config.py
git commit -m "feat: configure PC checkpoint retention"
```

### Task 2: Prune old PC checkpoints after a successful save

**Files:**
- Modify: `train_pc.py:1-100, 228-232`
- Modify: `tests/test_train_pc.py`

**Interfaces:**
- Produces: `prune_pc_checkpoints(root: str | Path, limit: int) -> None`.
- Consumes: a checkpoint root and positive retention limit; deletes only older numeric `checkpoint-<step>` directories.

- [ ] **Step 1: Add the failing pruning test.** Add `tempfile` is already imported; add:

```python
def test_prune_pc_checkpoints_keeps_only_latest_numeric_states():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for name in ("checkpoint-250", "checkpoint-500", "checkpoint-750", "checkpoint-draft"):
            (root / name).mkdir()

        prune_pc_checkpoints(root, 2)

        assert {path.name for path in root.iterdir()} == {
            "checkpoint-500", "checkpoint-750", "checkpoint-draft"
        }
```

- [ ] **Step 2: Run the pruning test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py::test_prune_pc_checkpoints_keeps_only_latest_numeric_states`

Expected: collection fails because `prune_pc_checkpoints` is not defined.

- [ ] **Step 3: Add the helper.** Import `shutil` and add:

```python
def prune_pc_checkpoints(root: str | Path, limit: int) -> None:
    """Keep only the newest numeric Accelerate PC checkpoint directories."""
    checkpoints = sorted(
        (
            path for path in Path(root).glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    for checkpoint in checkpoints[:-limit]:
        shutil.rmtree(checkpoint)
```

- [ ] **Step 4: Invoke pruning after successful saves.**

```python
if step % config["checkpointing_steps"] == 0:
    accelerator.save_state(output_dir / f"checkpoint-{step}")
    if accelerator.is_main_process:
        prune_pc_checkpoints(output_dir, config["checkpoints_total_limit"])
```

- [ ] **Step 5: Run PC trainer tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py`

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add train_pc.py tests/test_train_pc.py
git commit -m "feat: retain recent PC checkpoints"
```

### Task 3: Configure the Utonia Slurm run

**Files:**
- Modify: `submit_utonia.sh:35-39`
- Modify: `tests/test_submit_utonia.py`

**Interfaces:**
- Produces: an Utonia submission command with `max_train_steps=10000`, `checkpoints_total_limit=2`, and `resume_from_checkpoint=latest`.

- [ ] **Step 1: Extend the launcher contract test.**

```python
assert "max_train_steps=10000" in script
assert "checkpoints_total_limit=2" in script
assert "resume_from_checkpoint=latest" in script
```

- [ ] **Step 2: Run the launcher test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_submit_utonia.py`

Expected: FAIL because the two run-limit overrides are absent.

- [ ] **Step 3: Add the two overrides after the config argument.**

```bash
        --config configs/train/config_pc_utonia_overfit.yaml \
        max_train_steps=10000 \
        checkpoints_total_limit=2 \
        resume_from_checkpoint=latest
```

- [ ] **Step 4: Run launcher test and Bash syntax verification.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_submit_utonia.py && bash -n submit_utonia.sh`

Expected: PASS with no Bash syntax output.

- [ ] **Step 5: Commit.**

```bash
git add submit_utonia.sh tests/test_submit_utonia.py
git commit -m "feat: cap Utonia Slurm checkpoint retention"
```

### Task 4: Verify the completed integration

**Files:**
- Verify only; preserve unrelated worktree changes.

- [ ] **Step 1: Run focused retention tests.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py tests/test_train_pc.py tests/test_submit_utonia.py`

Expected: PASS.

- [ ] **Step 2: Run the full suite.**

Run: `conda run -n utonia-dev python -m pytest -q`

Expected: all tests pass; existing CPU-only CUDA-autocast warnings may remain.

- [ ] **Step 3: Check whitespace and status.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and no unrelated files staged.

- [ ] **Step 4: Report runtime behavior.**

Run: `sbatch submit_utonia.sh`

Expected: the Utonia job stops after 10,000 optimizer steps and keeps only `checkpoint-9750` and `checkpoint-10000` if it reaches the cap with the default 250-step save cadence.
