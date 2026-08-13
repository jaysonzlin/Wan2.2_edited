# Vendor Utonia Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the Utonia inference runtime into Wan so the point-cloud trainer and container no longer depend on a sibling Utonia checkout or editable install.

**Architecture:** Copy the Apache-2.0 Utonia inference subset into `wan.utonia` with its original relative-import structure. The existing feature extractor changes only its package import; it continues to load `Pointcept/Utonia` weights from Hugging Face and cache features/checkpoints exactly as before. The container retains Utonia's runtime wheels but removes source injection and packaging workarounds.

**Tech Stack:** Python 3.10, PyTorch 2.4/cu124, spconv-cu124, torch-scatter, timm, Hugging Face Hub, pytest, Apptainer/Singularity.

## Global Constraints

- Preserve all Utonia Apache-2.0 copyright/license notices and add a readable copy of Utonia's license under `wan/utonia/LICENSE`.
- Vendor only `__init__.py`, `data.py`, `model.py`, `module.py`, `registry.py`, `structure.py`, `transform.py`, `utils.py`, and `serialization/`; exclude `pca_trajectory.py`, `joint_trajectory_pca.py`, and `rgb_trajectory.py`.
- Do not commit model weights or change the `Pointcept/Utonia` Hugging Face repository, checkpoint cache root, RGB preprocessing, or feature cache format.
- Retain the existing Torch 2.4/cu124, `spconv-cu124`, `torch-scatter`, and precompiled Flash Attention declarations.
- `current.def` must not reference `../utonia`, `/opt/Utonia`, or an editable Utonia pip install after the change.

---

## File Map

| File | Responsibility |
| --- | --- |
| `wan/utonia/` | Vendored Utonia inference runtime with source attribution. |
| `wan/utonia/LICENSE` | Apache-2.0 license distributed with the vendored code. |
| `training/utonia_features.py` | Imports the vendored package for model loading and transforms. |
| `pyproject.toml` | Includes `wan.utonia` subpackages in installed distributions. |
| `current.def` | Retains runtime extensions but removes the sibling source copy and editable installation. |
| `tests/test_vendored_utonia.py` | Pins the source boundary, attribution, and packaging discovery. |
| `tests/test_utonia_features.py` | Proves the extractor resolves the vendored import seam. |
| `tests/test_current_def.py` | Pins the self-contained container declaration. |
| `README.md` | Documents that no separate Utonia installation is required. |

## Tasks

### Task 1: Vendor the inference-only Utonia source boundary

**Files:**
- Create: `wan/utonia/__init__.py`, `data.py`, `model.py`, `module.py`, `registry.py`, `structure.py`, `transform.py`, `utils.py`, `LICENSE`
- Create: `wan/utonia/serialization/__init__.py`, `default.py`, `hilbert.py`, `z_order.py`
- Create: `tests/test_vendored_utonia.py`

**Interfaces:**
- Consumes: the checked-out source files in `../utonia/utonia/` and `../utonia/LICENSE`.
- Produces: importable package namespace `wan.utonia` containing `model` and `transform` used by the trainer.

- [ ] **Step 1: Write the failing source-boundary test.**

```python
from pathlib import Path


def test_vendored_utonia_contains_only_the_inference_runtime():
    root = Path("wan/utonia")
    required = {
        "__init__.py", "data.py", "model.py", "module.py", "registry.py",
        "structure.py", "transform.py", "utils.py", "serialization/__init__.py",
        "serialization/default.py", "serialization/hilbert.py", "serialization/z_order.py",
        "LICENSE",
    }
    assert all((root / path).is_file() for path in required)
    assert "Apache License" in (root / "LICENSE").read_text()
    assert not (root / "pca_trajectory.py").exists()
    assert not (root / "joint_trajectory_pca.py").exists()
    assert not (root / "rgb_trajectory.py").exists()
```

- [ ] **Step 2: Run the test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_vendored_utonia.py`

Expected: FAIL because `wan/utonia` does not exist.

- [ ] **Step 3: Add the exact upstream inference files and license.** Preserve the existing source headers and copy byte-for-byte from the listed `../utonia/utonia/` paths. Do not copy build products, tests, `setup.py`, `.git`, model outputs, or the three excluded trajectory/PCA modules.

- [ ] **Step 4: Run the source-boundary test to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_vendored_utonia.py`

Expected: PASS.

- [ ] **Step 5: Commit the vendored source.**

```bash
git add wan/utonia tests/test_vendored_utonia.py
git commit -m "feat: vendor Utonia inference runtime"
```

### Task 2: Route the feature extractor through `wan.utonia`

**Files:**
- Modify: `training/utonia_features.py:228-252`
- Modify: `tests/test_utonia_features.py`

**Interfaces:**
- Consumes: `wan.utonia.model.load(name, repo_id, download_root, **kwargs)` and `wan.utonia.transform.default(...)`.
- Produces: `UtoniaFeatureExtractor(cache_root)` with the same `checkpoint_fingerprint`, preprocessing, and dense-feature behavior, without importing a top-level installed `utonia` package.

- [ ] **Step 1: Write a failing vendored-import seam test.** Add `types` and `wan` imports, then add:

```python
def test_extractor_uses_vendored_utonia(monkeypatch, tmp_path):
    checkpoint = tmp_path / "_utonia_checkpoint" / "utonia.pth"

    class FakeModel:
        def cuda(self): return self
        def eval(self): return self

    def load(*_args, **kwargs):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")
        assert kwargs["repo_id"] == "Pointcept/Utonia"
        return FakeModel()

    fake_utonia = types.SimpleNamespace(
        model=types.SimpleNamespace(load=load),
        transform=types.SimpleNamespace(default=lambda **_kwargs: object()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan, "utonia", fake_utonia, raising=False)

    extractor = UtoniaFeatureExtractor(tmp_path)

    assert extractor.model.__class__ is FakeModel
```

- [ ] **Step 2: Run the seam test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_utonia_features.py::test_extractor_uses_vendored_utonia`

Expected: FAIL because the extractor still imports top-level `utonia`.

- [ ] **Step 3: Replace the local `import utonia` with `from wan import utonia`.** Keep the lazy import, CUDA guard, weight loader arguments, transforms, and cache fingerprint unchanged.

- [ ] **Step 4: Run the seam and feature-cache tests.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_utonia_features.py`

Expected: PASS.

- [ ] **Step 5: Commit the import routing.**

```bash
git add training/utonia_features.py tests/test_utonia_features.py
git commit -m "feat: use vendored Utonia extractor"
```

### Task 3: Make Wan package discovery include the vendored subpackages

**Files:**
- Modify: `pyproject.toml:42-46`
- Modify: `tests/test_vendored_utonia.py`

**Interfaces:**
- Consumes: `pyproject.toml` setuptools configuration.
- Produces: source distributions and editable installs that include `wan`, `wan.utonia`, and `wan.utonia.serialization`.

- [ ] **Step 1: Extend the failing source-layout test.**

```python
def test_packaging_discovers_wan_subpackages():
    definition = Path("pyproject.toml").read_text()
    assert '[tool.setuptools.packages.find]' in definition
    assert 'include = ["wan", "wan.*"]' in definition
```

- [ ] **Step 2: Run it to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_vendored_utonia.py::test_packaging_discovers_wan_subpackages`

Expected: FAIL because the project currently lists only `packages = ["wan"]`.

- [ ] **Step 3: Replace the explicit top-level packages list.**

```toml
[tool.setuptools.packages.find]
include = ["wan", "wan.*"]
```

- [ ] **Step 4: Run the vendored-package tests.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_vendored_utonia.py`

Expected: PASS.

- [ ] **Step 5: Commit packaging discovery.**

```bash
git add pyproject.toml tests/test_vendored_utonia.py
git commit -m "build: package vendored Utonia modules"
```

### Task 4: Remove the external Utonia container dependency and update usage docs

**Files:**
- Modify: `current.def:4-7, 62-64, 98-102`
- Modify: `tests/test_current_def.py`
- Modify: `README.md:40-53`

**Interfaces:**
- Consumes: the in-repository `wan.utonia` package and existing binary Utonia runtime dependencies.
- Produces: a container definition buildable from this repository alone and documentation with no separate Utonia installation step.

- [ ] **Step 1: Write the failing self-contained-container assertions.** Replace the external-install assertions with:

```python
assert "../utonia /opt/Utonia" not in definition
assert "/opt/Utonia" not in definition
assert "pip install --no-deps --no-build-isolation -e" not in definition
assert 'pip install --upgrade pip setuptools wheel' in definition
```

Add a README assertion that the Utonia overfit section says the runtime is vendored and does not instruct the user to install a sibling Utonia package.

- [ ] **Step 2: Run the focused tests to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_current_def.py tests/test_train_pc.py`

Expected: FAIL because `current.def` still copies and installs `/opt/Utonia`, and README still requires a sibling package.

- [ ] **Step 3: Remove only the external-source plumbing.** Delete the `%files` Utonia injection and its editable install. Restore the general package bootstrap command:

```bash
$ENV_BIN/pip install --upgrade pip setuptools wheel
```

Keep the cu124 binary extension declarations, Utonia runtime requirements, Flash Attention URL, Miniforge downloader, and weight-download behavior unchanged. Update the README section to state that Utonia inference code is vendored while weights are fetched on first use.

- [ ] **Step 4: Run focused tests to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_current_def.py tests/test_train_pc.py`

Expected: PASS.

- [ ] **Step 5: Commit the self-contained container integration.**

```bash
git add current.def tests/test_current_def.py README.md tests/test_train_pc.py
git commit -m "build: remove external Utonia install"
```

### Task 5: Verify the completed integration

**Files:**
- Verify only; preserve unrelated dirty worktree files.

- [ ] **Step 1: Run whitespace validation.**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 2: Run the full repository suite.**

Run: `conda run -n utonia-dev python -m pytest -q`

Expected: all tests pass; existing CPU-only CUDA-autocast warnings may remain.

- [ ] **Step 3: Check the final source boundary.**

Run: `rg -n "\.\./utonia|/opt/Utonia|pip install .* -e .*Utonia" current.def README.md tests`

Expected: no matches.

- [ ] **Step 4: Report the build boundary.** State that `apptainer build --fakeroot wan-utonia.sif current.def` now requires only this repository checkout. A successful GPU-enabled feature-cache run verifies Utonia model execution and the Hugging Face download path.
