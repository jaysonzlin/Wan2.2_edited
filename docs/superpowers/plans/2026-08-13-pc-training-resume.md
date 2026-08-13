# PC Training Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train_pc.py` load an Accelerate checkpoint selected by `resume_from_checkpoint` and continue from its saved global step.

**Architecture:** Add small checkpoint-selection helpers to `train_pc.py`, matching the existing I2V trainer’s numeric `checkpoint-<step>` convention and latest-fallback behavior. Invoke the helper after `accelerator.prepare(...)`, then seed the progress bar and training loop with the restored step. Validate the PC config’s resume setting before model setup.

**Tech Stack:** Python 3.10, Accelerate state checkpoints, pytest.

## Global Constraints

- Preserve existing checkpoint names (`checkpoint-<step>`), save cadence, Utonia cache behavior, and output layout.
- Accept only `null`, `"latest"`, or a non-empty explicit path for `resume_from_checkpoint`.
- For `"latest"`, try numeric checkpoint directories newest-first and skip incomplete checkpoints only when a later candidate exists.
- For an explicit path, propagate `accelerator.load_state(...)` failures instead of falling back.
- Resume after `accelerator.prepare(...)`; use the selected directory’s numeric suffix as `step` and progress-bar initial position.
- Do not add intra-epoch dataloader-position restore.

---

## File Map

| File | Responsibility |
| --- | --- |
| `training/pc_config.py` | Validates accepted resume settings. |
| `train_pc.py` | Selects/loads Accelerate state and restores global step. |
| `tests/test_pc_config.py` | Pins resume-setting validation. |
| `tests/test_train_pc.py` | Tests latest selection, fallback, explicit paths, diagnostics, and restored step plumbing. |

## Tasks

### Task 1: Validate PC resume configuration

**Files:**
- Modify: `training/pc_config.py:50-90`
- Modify: `tests/test_pc_config.py`

**Interfaces:**
- Consumes: top-level PC config field `resume_from_checkpoint`.
- Produces: `load_pc_config(path, overrides)` that accepts `None`, `"latest"`, and non-empty strings; otherwise raises `ValueError`.

- [ ] **Step 1: Write the failing validation tests.**

```python
@pytest.mark.parametrize("resume", ["latest", "outputs/run/checkpoint-250"])
def test_pc_config_accepts_resume_setting(tmp_path, resume):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text() + f"resume_from_checkpoint: {resume}\n")
    assert load_pc_config(path, [])["resume_from_checkpoint"] == resume


@pytest.mark.parametrize("resume", ["0", "[]", "'   '"])
def test_pc_config_rejects_invalid_resume_setting(tmp_path, resume):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text() + f"resume_from_checkpoint: {resume}\n")
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        load_pc_config(path, [])
```

- [ ] **Step 2: Run the new validation tests to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py`

Expected: the invalid resume cases fail because validation does not yet reject them.

- [ ] **Step 3: Add the minimal validation.** At the end of `validate_pc_config`, add:

```python
resume = config.get("resume_from_checkpoint")
if resume is not None and (not isinstance(resume, str) or not resume.strip()):
    raise ValueError("resume_from_checkpoint must be null, 'latest', or a non-empty path string")
```

- [ ] **Step 4: Run the PC-config tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add training/pc_config.py tests/test_pc_config.py
git commit -m "feat: validate PC resume setting"
```

### Task 2: Add checkpoint selection and fallback helpers

**Files:**
- Modify: `train_pc.py:1-40`
- Modify: `tests/test_train_pc.py`

**Interfaces:**
- Produces: `_pc_checkpoint_paths(output_dir: Path, setting: str | None) -> list[Path]` and `load_pc_checkpoint_with_fallback(accelerator, output_dir: Path, setting: str | None) -> Path | None`.
- Consumes: checkpoint directories named `checkpoint-<integer>` and an accelerator exposing `load_state(path)`.

- [ ] **Step 1: Write the failing helper tests.** Add `tempfile` and the helper import, then add:

```python
def test_pc_latest_checkpoint_falls_back_after_a_failed_load():
    class FakeAccelerator:
        def __init__(self): self.attempts = []
        def load_state(self, path):
            self.attempts.append(Path(path).name)
            if Path(path).name == "checkpoint-750":
                raise RuntimeError("incomplete checkpoint")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for step in (250, 500, 750): (root / f"checkpoint-{step}").mkdir()
        accelerator = FakeAccelerator()
        resumed = load_pc_checkpoint_with_fallback(accelerator, root, "latest")

    assert resumed.name == "checkpoint-500"
    assert accelerator.attempts == ["checkpoint-750", "checkpoint-500"]


def test_pc_explicit_checkpoint_propagates_load_failure(tmp_path):
    checkpoint = tmp_path / "checkpoint-250"
    checkpoint.mkdir()

    class FakeAccelerator:
        def load_state(self, _path): raise RuntimeError("incomplete checkpoint")

    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        load_pc_checkpoint_with_fallback(FakeAccelerator(), tmp_path, str(checkpoint))


def test_pc_latest_checkpoint_reports_all_failed_candidates(tmp_path):
    for step in (250, 500): (tmp_path / f"checkpoint-{step}").mkdir()

    class FakeAccelerator:
        def load_state(self, path): raise RuntimeError(f"incomplete {Path(path).name}")

    with pytest.raises(RuntimeError, match="checkpoint-500, checkpoint-250"):
        load_pc_checkpoint_with_fallback(FakeAccelerator(), tmp_path, "latest")
```

- [ ] **Step 2: Run the helper tests to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py::test_pc_latest_checkpoint_falls_back_after_a_failed_load tests/test_train_pc.py::test_pc_explicit_checkpoint_propagates_load_failure tests/test_train_pc.py::test_pc_latest_checkpoint_reports_all_failed_candidates`

Expected: collection fails because `load_pc_checkpoint_with_fallback` is not defined.

- [ ] **Step 3: Add the checkpoint helpers.**

```python
def _pc_checkpoint_paths(output_dir: Path, setting: str | None) -> list[Path]:
    if not setting:
        return []
    if setting != "latest":
        return [Path(setting)]
    return sorted(
        (path for path in output_dir.glob("checkpoint-*")
         if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )


def load_pc_checkpoint_with_fallback(accelerator, output_dir: Path, setting: str | None) -> Path | None:
    checkpoints = _pc_checkpoint_paths(output_dir, setting)
    if not checkpoints:
        return None
    failures = []
    for checkpoint in checkpoints:
        try:
            accelerator.load_state(checkpoint)
        except Exception as error:
            if setting != "latest":
                raise
            failures.append((checkpoint, error))
            print(f"Could not load {checkpoint}; trying the next most recent checkpoint: {error}")
        else:
            return checkpoint
    attempted = ", ".join(path.name for path, _ in failures)
    raise RuntimeError(
        f"Could not load any checkpoint selected by resume_from_checkpoint=latest: {attempted}"
    ) from failures[-1][1]
```

- [ ] **Step 4: Run the helper tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py::test_pc_latest_checkpoint_falls_back_after_a_failed_load tests/test_train_pc.py::test_pc_explicit_checkpoint_propagates_load_failure tests/test_train_pc.py::test_pc_latest_checkpoint_reports_all_failed_candidates`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add train_pc.py tests/test_train_pc.py
git commit -m "feat: add PC checkpoint resume helpers"
```

### Task 3: Restore the PC global step after Accelerate preparation

**Files:**
- Modify: `train_pc.py:123-147`
- Modify: `tests/test_train_pc.py`

**Interfaces:**
- Consumes: `load_pc_checkpoint_with_fallback(accelerator, output_dir, config.get("resume_from_checkpoint")) -> Path | None`.
- Produces: progress bar initialized at the loaded checkpoint’s numeric step; all prepared Accelerate state is restored before training resumes.

- [ ] **Step 1: Write the failing source-level regression test.**

```python
def test_train_pc_restores_step_after_accelerator_prepare():
    source = Path("train_pc.py").read_text()
    prepared = source.index("accelerator.prepare")
    restored = source.index("load_pc_checkpoint_with_fallback")
    assert restored > prepared
    assert 'step = int(resume_path.name.removeprefix("checkpoint-"))' in source
    assert "initial=step" in source
```

- [ ] **Step 2: Run the regression test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py::test_train_pc_restores_step_after_accelerator_prepare`

Expected: FAIL because the trainer currently sets `step = 0` unconditionally.

- [ ] **Step 3: Load state and restore `step`.** Directly after `accelerator.prepare(...)`, replace the unconditional initialization with:

```python
resume_path = load_pc_checkpoint_with_fallback(
    accelerator, output_dir, config.get("resume_from_checkpoint")
)
step = int(resume_path.name.removeprefix("checkpoint-")) if resume_path else 0
```

Leave `create_progress_bar(..., initial=step, ...)` unchanged.

- [ ] **Step 4: Run the PC trainer tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_train_pc.py`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add train_pc.py tests/test_train_pc.py
git commit -m "feat: resume PC training state"
```

### Task 4: Verify integration

**Files:**
- Verify only; leave unrelated working-tree changes untouched.

- [ ] **Step 1: Run focused resume tests.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_pc_config.py tests/test_train_pc.py tests/test_submit_utonia.py`

Expected: PASS. The launcher’s `training.resume_from_checkpoint=latest` override is now functional.

- [ ] **Step 2: Run the complete suite.**

Run: `conda run -n utonia-dev python -m pytest -q`

Expected: all tests pass; existing CPU-only CUDA-autocast warnings may remain.

- [ ] **Step 3: Check whitespace and status.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and no unrelated files staged.

- [ ] **Step 4: Report runtime use.**

Run: `sbatch submit_utonia.sh`

Expected: a requeued job loads the newest complete checkpoint under `outputs/pc_trajectory_utonia_overfit/` before continuing its optimizer steps.
