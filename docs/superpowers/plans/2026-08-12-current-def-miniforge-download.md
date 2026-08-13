# Current Definition Miniforge Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `current.def` download and remove its Miniforge installer during `%post`, eliminating the external installer build-context input.

**Architecture:** Retain the existing system dependency and Miniforge installation order, but replace the `%files` copy with Utonia's `wget -qO /tmp/miniforge.sh` lifecycle. A static pytest test asserts the exact download/install/cleanup contract without building a container.

**Tech Stack:** Apptainer/Singularity definition syntax, shell, pytest.

## Global Constraints

- Download the official x86_64 Miniforge installer from the conda-forge `latest/download` URL.
- Install it to `/opt/conda` and delete `/tmp/miniforge.sh` in `%post`.
- Do not change Python, CUDA, Utonia, or runtime dependency pins.
- Do not retain a `%files` reference to `./Miniforge3-Linux-x86_64.sh`.

---

## File map

| File | Responsibility |
| --- | --- |
| `current.def` | Download, install, and remove Miniforge as part of the container build. |
| `tests/test_current_def.py` | Static validation of the definition's Miniforge and Utonia dependency declarations. |

## Tasks

### Task 1: Replace injected Miniforge with a download lifecycle

**Files:**
- Modify: `tests/test_current_def.py`
- Modify: `current.def`

**Interfaces:**
- Consumes: repository-root `current.def` text.
- Produces: a build definition that needs no local Miniforge installer file.

- [ ] **Step 1: Write the failing static assertions.** Extend the current definition test with assertions for the official `wget` URL, `/tmp/miniforge.sh` install/cleanup, and absence of the old `%files` installer entry.

```python
assert "wget -qO /tmp/miniforge.sh" in definition
assert "bash /tmp/miniforge.sh -b -p /opt/conda" in definition
assert "rm /tmp/miniforge.sh" in definition
assert "./Miniforge3-Linux-x86_64.sh /opt/miniforge.sh" not in definition
```

- [ ] **Step 2: Run the static test to verify red.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: fails because the current definition still copies and installs `/opt/miniforge.sh`.

- [ ] **Step 3: Implement the minimal definition change.** Remove only the Miniforge `%files` comment/entry. Replace the installation block with:

```bash
wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p /opt/conda
rm /tmp/miniforge.sh
```

- [ ] **Step 4: Run the static test to verify green.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: passes without downloading Miniforge or creating a container.

- [ ] **Step 5: Commit.**

```bash
git add current.def tests/test_current_def.py
git commit -m "build: download Miniforge in container definition"
```

### Task 2: Verify the repository state

**Files:**
- Verify only; do not alter unrelated dirty files.

- [ ] **Step 1: Run the full Python suite.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q`

Expected: all repository tests pass; existing CPU CUDA-autocast warnings may remain.

- [ ] **Step 2: Check whitespace and status.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; unrelated user-owned worktree files remain untouched.
