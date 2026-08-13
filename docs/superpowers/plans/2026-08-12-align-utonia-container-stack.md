# Align Utonia Container Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Utonia's proven Torch 2.4/cu124 extension stack in `current.def` so `torch-scatter` resolves from a matching binary wheel.

**Architecture:** The CUDA 12.9 base and application requirements remain intact. Only the PyTorch index/version, PyG wheel URL, and spconv package are changed to Utonia's established cu124 trio; the precompiled FlashAttention URL remains unchanged.

**Tech Stack:** Apptainer/Singularity definition syntax, Python 3.10, PyTorch 2.4.1/cu124, PyG `torch-scatter`, spconv-cu124, pytest.

## Global Constraints

- Retain `nvidia/cuda:12.9.2-devel-ubuntu22.04`, the local Utonia injection, and the existing precompiled FlashAttention URL.
- Install `torch==2.4.1` and `torchvision==0.19.1` using `https://download.pytorch.org/whl/cu124`.
- Install `torch-scatter` from `https://data.pyg.org/whl/torch-2.4.0+cu124.html`.
- Install `spconv-cu124` rather than `spconv-cu126`.
- Do not change the runtime model-weight download/cache behavior.

---

## File map

| File | Responsibility |
| --- | --- |
| `current.def` | Defines the aligned Utonia Torch, PyG, and spconv binary package set. |
| `tests/test_current_def.py` | Statically pins the exact compatible declaration set. |

## Tasks

### Task 1: Pin the Utonia-compatible binary extension stack

**Files:**
- Modify: `tests/test_current_def.py`
- Modify: `current.def`

**Interfaces:**
- Consumes: repository-root `current.def` text.
- Produces: a build definition that requests binary extension wheels built for Torch 2.4/cu124.

- [x] **Step 1: Write the failing static assertions.** Replace the existing 2.5.1/cu126 expectations with exact assertions for Torch 2.4.1, torchvision 0.19.1, PyTorch cu124, PyG `torch-2.4.0+cu124.html`, and `spconv-cu124`. Assert that the old 2.5.1/cu126 extension declarations are absent and the FlashAttention URL remains present.

```python
assert '"torch==2.4.1" "torchvision==0.19.1"' in definition
assert "https://download.pytorch.org/whl/cu124" in definition
assert "torch-2.4.0+cu124.html" in definition
assert "spconv-cu124" in definition
assert "spconv-cu126" not in definition
```

- [x] **Step 2: Run the static test to verify red.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: fails because `current.def` still declares Torch 2.5.1/cu126 and `spconv-cu126`.

- [x] **Step 3: Apply the three definition changes.** Replace only these declarations:

```bash
$ENV_BIN/pip install "numpy>=1.23.5,<2" "torch==2.4.1" "torchvision==0.19.1" --extra-index-url https://download.pytorch.org/whl/cu124
$ENV_BIN/pip install --no-cache-dir spconv-cu124 torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Leave the local Utonia install and `flash_attn-2.6.3+cu126torch2.4-...` URL unchanged.

- [x] **Step 4: Run the static test to verify green.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: passes without building a container or downloading packages.

- [x] **Step 5: Commit.**

```bash
git add current.def tests/test_current_def.py
git commit -m "build: align Utonia extension stack"
```

### Task 2: Verify repository state and state the container boundary

**Files:**
- Verify only; do not alter unrelated dirty files.

- [x] **Step 1: Run the full suite.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q`

Expected: all repository tests pass; existing CPU CUDA-autocast warnings may remain.

- [x] **Step 2: Check whitespace and worktree status.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; user-owned dirty files remain untouched.

- [x] **Step 3: Report integration verification.** State that `apptainer build --fakeroot wan-utonia.sif current.def` must run from the repository root with `../utonia` available. The successful build verifies that pip selects the binary PyG wheel instead of the source tarball; a GPU-enabled run verifies actual Utonia feature extraction.
