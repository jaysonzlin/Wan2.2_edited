# Utonia Singularity Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `current.def` build a CUDA 12.9 image containing the local Utonia package and CUDA extensions compatible with its Torch 2.5.1/cu126 environment.

**Architecture:** The definition injects the sibling Utonia source under `/opt/Utonia`, installs Torch and compiled extension wheels in an explicit order, then builds FlashAttention from source against the installed Torch. A build-time import probe checks that all modules required by `UtoniaFeatureExtractor` load before the image is produced.

**Tech Stack:** Apptainer/Singularity definition file, CUDA 12.9, Python 3.10, PyTorch 2.5.1/cu126, PyG wheels, spconv, FlashAttention, Utonia.

## Global Constraints

- Retain `nvidia/cuda:12.9.2-devel-ubuntu22.04`, Python 3.10, Torch 2.5.1, and torchvision 0.20.1.
- Install Torch from `https://download.pytorch.org/whl/cu126` and `torch-scatter` from `https://data.pyg.org/whl/torch-2.5.1+cu126.html`.
- Install `spconv-cu126==2.3.8`, `timm`, and `addict`.
- Copy the local sibling `../Utonia` source to `/opt/Utonia`; do not clone its code during `%post`.
- Replace the incompatible Torch-2.4 FlashAttention binary with `flash-attn==2.6.3 --no-build-isolation`.
- Do not bake Hugging Face model weights into the image; runtime cache preparation downloads `Pointcept/Utonia` as before.

---

## File map

| File | Responsibility |
| --- | --- |
| `current.def` | Defines all container build inputs, Utonia source injection, matching dependency installation, and import verification. |
| `tests/test_current_def.py` (new) | Static regression test for required pins/install instructions; it does not build a container. |

## Tasks

### Task 1: Add a static contract test for the container definition

**Files:**
- Create: `tests/test_current_def.py`
- Modify: `current.def`

**Interfaces:**
- Consumes: repository-root `current.def` text.
- Produces: a CPU-only regression check that detects an incompatible Utonia/FlashAttention dependency declaration before a costly image build.

- [ ] **Step 1: Write the failing test.** Assert the definition injects `/opt/Utonia`, uses the Torch 2.5.1/cu126 and matching PyG wheel URLs, installs `spconv-cu126==2.3.8`, installs local Utonia editable, builds `flash-attn==2.6.3 --no-build-isolation`, and includes a Python import probe. Assert it does not contain `torch2.4-cp310`.

```python
def test_current_def_declares_a_matching_utonia_runtime():
    definition = Path("current.def").read_text()

    assert "../Utonia /opt/Utonia" in definition
    assert "torch-2.5.1+cu126.html" in definition
    assert "spconv-cu126==2.3.8" in definition
    assert "flash-attn==2.6.3 --no-build-isolation" in definition
    assert "torch2.4-cp310" not in definition
```

- [ ] **Step 2: Run the test to verify red.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: fails because the existing definition has no Utonia injection/dependencies and contains the Torch-2.4 FlashAttention wheel.

- [ ] **Step 3: Add the Utonia source and dependencies.** In `%files`, add `../Utonia /opt/Utonia` after the Miniforge installer. Use the cu126 PyTorch index as the primary index, then install:

```bash
$ENV_BIN/pip install --no-cache-dir \
  "spconv-cu126==2.3.8" \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.5.1+cu126.html
$ENV_BIN/pip install --no-cache-dir timm addict psutil einops
$ENV_BIN/pip install --no-deps -e /opt/Utonia
$ENV_BIN/pip install --no-build-isolation "flash-attn==2.6.3"
```

Remove the old direct prebuilt FlashAttention URL. Keep cleanup after all package installation.

- [ ] **Step 4: Add the build-time import probe.** Immediately after the install commands, add:

```bash
$ENV_BIN/python - <<'PY'
import flash_attn
import spconv.pytorch
import torch
import torch_scatter
import utonia

assert torch.__version__.startswith("2.5.1")
assert torch.version.cuda == "12.6"
print(f"Utonia runtime ready: torch={torch.__version__}, cuda={torch.version.cuda}")
PY
```

- [ ] **Step 5: Run the static test to verify green.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q tests/test_current_def.py`

Expected: passes without downloading packages or building an image.

- [ ] **Step 6: Commit.**

```bash
git add current.def tests/test_current_def.py
git commit -m "build: add Utonia to Singularity runtime"
```

### Task 2: Verify repository and provide the GPU build boundary

**Files:**
- Verify only; no changes to unrelated dirty files.

- [ ] **Step 1: Run the full Python suite.**

Run: `conda run -n utonia-dev --cwd /Users/jaysonlin/Desktop/Current/Wan2.2_edited python -m pytest -q`

Expected: all repository tests pass; existing CPU CUDA-autocast warnings may remain.

- [ ] **Step 2: Check the resulting worktree.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and no modifications to user-owned dirty files.

- [ ] **Step 3: Report the external build/launch requirement.** Provide the build command from the repository root, such as `apptainer build --fakeroot wan-utonia.sif current.def`. State that the build context must contain `Miniforge3-Linux-x86_64.sh` and its sibling `../Utonia` checkout. State that actual feature extraction still needs a GPU-enabled runtime and network or a mounted persistent Utonia cache for the official checkpoint download.
