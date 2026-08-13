# Utonia PC Trajectory Overfit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive, single-GPU `train_pc.py` experiment that overfits object `000` while conditioning every trajectory point token on frozen, dense Utonia features from frame-zero XYZ and RGB.

**Architecture:** The trainer resolves object-level HDF5 inputs and builds/reuses a separate validated Utonia cache before workers or the optimizer exist. The dataset subsequently loads dense features from that cache. `PCTrajectoryModel` fuses the per-point condition into coordinate tokens immediately after `PointEmbed`, and both sampling pipelines forward the same condition for visualizations.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, HDF5/h5py, Hugging Face Hub, the sibling Utonia/PTv3 package, pytest.

## Global constraints

- Preserve coordinate-only `config_pc.yaml`, its dataset layout, and all joint Wan--PhysCtrl trainers unchanged.
- The new dataset mode reads `sample_*/objects/000/pc.hdf5`. It requires `rgb: uint8 [2048, 3]`; do not read normals from HDF5, and provide Utonia zero normals.
- Build the cache once in the main process before DataLoader construction. Cached tensors are float32 `[2048, D]`; workers only read validated cache files.
- Download/load official weights through Utonia's Hugging Face path for `Pointcept/Utonia`; use `eval()` and no gradients, then release the extractor before training.
- Preserve Utonia's documented dense-feature upcast and inverse mapping. Make its nominally train-mode grid sample reproducible by deriving and scoping a fixed NumPy seed from the frame-zero source fingerprint.
- Do not hard-code `D`; infer it from the first valid cache entry, require all entries to match, and construct the model/optimizer only afterward.
- Follow normal PyTorch initialization for the new `LayerNorm` and `Linear`; do not add a special identity or zero initialization.

---

## File map

| File | Responsibility |
| --- | --- |
| `training/pc_dataset.py` | Add selected-object discovery, RGB validation, and optional cached-feature loading without altering baseline behavior. |
| `training/utonia_features.py` (new) | Deterministic Utonia extraction, cache metadata/fingerprints, validation, and cache preparation. |
| `training/pc_config.py` | Validate the enabled Utonia configuration contract. |
| `wan/modules/pc_trajectory.py` | Add optional post-`PointEmbed` LayerNorm + concat + Linear fusion. |
| `wan/pc_pipeline.py` | Carry optional Utonia features through DDIM and flow inference. |
| `train_pc.py` | Prepare cache before the DataLoader/model and pass condition during train and visualization. |
| `configs/train/config_pc_utonia_overfit.yaml` (new) | Dedicated one-object DDPM/x0 overfit experiment. |
| `tests/test_pc_dataset.py`, `tests/test_pc_config.py`, `tests/test_pc_trajectory_model.py`, `tests/test_pc_pipeline.py`, `tests/test_utonia_features.py` (new) | CPU regression coverage using fake extraction rather than a CUDA Utonia dependency. |
| `README.md` | Add concise setup/launch instructions for the Utonia dependency and first-run Hugging Face download. |

## Tasks

### 1. Define the enabled-config and selected-object dataset contract

**Files:**
- Modify: `training/pc_config.py`
- Modify: `training/pc_dataset.py`
- Modify: `tests/test_pc_config.py`
- Modify: `tests/test_pc_dataset.py`

- [ ] **Step 1: Write failing config tests.** Add tests proving the existing baseline config remains valid, while a Utonia-enabled config rejects a missing/empty `data.object_id` or `data.utonia_cache_root`. Add a test that the enabled flag is boolean and that baseline configuration does not need either Utonia field.

- [ ] **Step 2: Write failing selected-object/RGB tests.** Extend the fixture helpers to create `sample_*/objects/000/pc.hdf5`. Test that object mode returns the existing trajectory fields plus the cache condition once configured later, that missing `objects/000` fails during dataset construction, and that missing, non-`uint8`, wrong-shape, or invalid-range RGB raises a targeted `ValueError`. Test baseline `sample_*/pc.hdf5` discovery remains unchanged.

- [ ] **Step 3: Implement narrowly.** Extend `validate_pc_config` only when `model.utonia_enabled is True`; validate both required data keys and reject non-boolean enabled values. Extend `PCTrajectoryDataset` with explicit keyword-only object-mode/cache arguments. In object mode, enumerate only `sample_*/objects/<object_id>/pc.hdf5`, retain stable sample identities, validate frame-zero RGB against the fixed point count, and retain source paths for cache lookup. Keep its old one-argument construction behavior exactly intact.

- [ ] **Step 4: Run focused tests.**

Run: `pytest -q tests/test_pc_config.py tests/test_pc_dataset.py`

Expected: all configuration and dataset tests pass, including unchanged baseline discovery.

- [ ] **Step 5: Commit.**

```bash
git add training/pc_config.py training/pc_dataset.py tests/test_pc_config.py tests/test_pc_dataset.py
git commit -m "feat: validate Utonia PC object inputs"
```

### 2. Implement deterministic, validated Utonia feature caching

**Files:**
- Create: `training/utonia_features.py`
- Create: `tests/test_utonia_features.py`
- Modify: `training/pc_dataset.py`
- Modify: `tests/test_pc_dataset.py`

- [ ] **Step 1: Write CPU-only failing cache tests.** Use a fake extractor with a call counter and known per-point outputs. Cover: first-run creation, reuse without another extraction, rebuild after a changed frame-zero XYZ/RGB fingerprint, rebuild after changed checkpoint fingerprint or preprocessing version, corrupt/wrong-shape feature rejection, and mismatch across cached feature widths. Assert round-tripped features retain original `[2048, D]` point ordering. Do not import Utonia or require CUDA in these tests.

- [ ] **Step 2: Implement cache format and pure helpers.** Add a versioned cache record, written atomically, containing float32 contiguous features plus source identity/fingerprint (frame-zero XYZ and RGB), Utonia checkpoint fingerprint, preprocessing descriptor/version, point count, and feature dimension. Key files below the separate cache root by a stable sample/object identity; never write to HDF5. Provide pure validation/load helpers and a `prepare_utonia_feature_cache(...)` entry point that returns the common `D` only after all selected objects validate.

- [ ] **Step 3: Implement the real, lazy extractor.** Delay all Utonia/Hugging Face imports until preparation. Retrieve the official `Pointcept/Utonia` checkpoint using Utonia's Hugging Face workflow, hash the resolved checkpoint bytes, load the model through Utonia's loader, switch it to eval/no-grad, and free it after preparation. Feed only frame-zero coordinate and RGB (still uint8 until Utonia's color normalization) plus zero normal. Use the documented preprocessing (`normalize_coord=True`, fixed scale, center shift, grid coordinates, inverse, color normalization) with a source-derived scoped NumPy seed so GridSample's train-mode representative selection is repeatable. Reproduce Utonia's documented pooling upcast and use its final `inverse` mapping to restore dense original-point order.

- [ ] **Step 4: Wire cache reads into object-mode dataset.** Require a valid cache entry when feature mode is enabled; load it with CPU-safe `torch.load(..., weights_only=True)` (with a compatibility fallback if the repository's supported Torch version needs it), validate `[2048, D]`, and return it as `utonia_features`. Do not make worker processes build or update caches.

- [ ] **Step 5: Run focused tests.**

Run: `pytest -q tests/test_utonia_features.py tests/test_pc_dataset.py`

Expected: fake extraction verifies cache reuse/staleness/order and data workers only consume validated cache records.

- [ ] **Step 6: Commit.**

```bash
git add training/utonia_features.py training/pc_dataset.py tests/test_utonia_features.py tests/test_pc_dataset.py
git commit -m "feat: cache dense Utonia point features"
```

### 3. Add optional post-embedding Utonia fusion to the trajectory model

**Files:**
- Modify: `wan/modules/pc_trajectory.py`
- Modify: `tests/test_pc_trajectory_model.py`

- [ ] **Step 1: Write failing model tests.** Build small CPU models both with and without a supplied `utonia_feature_dim`. Verify coordinate-only calls preserve the present API and output shape. Verify configured fusion accepts `[B, N, D]`, broadcasts it over all 49 states, produces the expected output shape, and changes output when condition values change under a fixed seed. Assert descriptive `ValueError`s for missing feature on configured fusion, wrong batch size, wrong point count, and wrong feature width.

- [ ] **Step 2: Implement optional modules and validation.** Add an optional constructor argument `utonia_feature_dim: int | None = None`. When set, create `nn.LayerNorm(D)` and `nn.Linear(256 + D, 256)` using ordinary module construction. Thread optional `utonia_features` through `forward` and `encode_states`; after the existing coordinate `PointEmbed`, normalize features, expand them to `[B, 49, N, D]`, concatenate, project, then apply the existing fixed position embedding and all existing conditioning/blocks. When `D` is absent, do not instantiate fusion parameters and retain coordinate-only execution.

- [ ] **Step 3: Run focused tests.**

Run: `pytest -q tests/test_pc_trajectory_model.py`

Expected: both baseline and feature-conditioned CPU paths pass, including shape errors.

- [ ] **Step 4: Commit.**

```bash
git add wan/modules/pc_trajectory.py tests/test_pc_trajectory_model.py
git commit -m "feat: fuse dense Utonia features into PC tokens"
```

### 4. Forward the condition through DDIM and flow sampling

**Files:**
- Modify: `wan/pc_pipeline.py`
- Modify: `tests/test_pc_pipeline.py`

- [ ] **Step 1: Write failing forwarding tests.** Extend the fake trajectory model to record its optional feature argument. For both `PCDDIMPipeline` and `PCFlowPipeline`, assert the supplied `[B, N, D]` tensor reaches every model invocation unchanged; also assert existing no-feature callers still work.

- [ ] **Step 2: Implement backward-compatible pipeline arguments.** Add `utonia_features: torch.Tensor | None = None` as an optional keyword argument to both public `__call__` methods and pass it as a named argument to the model. Leave scheduler math, generator behavior, and current positional-call compatibility untouched.

- [ ] **Step 3: Run focused tests.**

Run: `pytest -q tests/test_pc_pipeline.py`

Expected: pipeline tests pass for DDIM, flow, conditioned, and coordinate-only inference.

- [ ] **Step 4: Commit.**

```bash
git add wan/pc_pipeline.py tests/test_pc_pipeline.py
git commit -m "feat: condition PC samplers on Utonia features"
```

### 5. Integrate the dedicated overfit trainer and configuration

**Files:**
- Create: `configs/train/config_pc_utonia_overfit.yaml`
- Modify: `train_pc.py`
- Modify: `tests/test_train_pc.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing trainer/config tests.** Mock cache preparation and the trajectory model in `tests/test_train_pc.py`. Assert the Utonia path prepares caches before DataLoader/model/optimizer construction, creates the model with the discovered feature width, and passes batch features during the train step and periodic visualization pipeline call. Assert baseline config takes no cache/extractor path. Add a load test for the dedicated YAML.

- [ ] **Step 2: Add the dedicated YAML.** Copy only the compatible baseline PC settings: 49 frames, 2048 points, 8 layers / width 256 / 4 heads, DDPM/x0, batch size one, and the existing visualization cadence. Set `model.utonia_enabled: true`, `data.object_id: "000"`, and a distinct `data.utonia_cache_root` outside source data. Do not edit `config_pc.yaml`.

- [ ] **Step 3: Integrate startup in `train_pc.py`.** For enabled mode: construct selected-object dataset metadata, call cache preparation on the main process before DataLoader creation, obtain/validate a single `D`, create the cache-backed dataset, then instantiate the model with `utonia_feature_dim=D` before constructing the optimizer. Pass `batch["utonia_features"]` to the model and the unwrapped visualization pipeline. Keep all baseline control flow and positional inputs untouched. Keep Utonia cache extraction outside Accelerate preparation and ensure the extractor reference is released before training.

- [ ] **Step 4: Document the manual integration boundary.** Add a short README section with the new command, state that the sibling Utonia package and its CUDA dependencies must be installed, and state that the first run obtains official Hugging Face weights and fills the separate cache root. Do not claim CPU-only Utonia inference support.

- [ ] **Step 5: Run focused tests.**

Run: `pytest -q tests/test_pc_config.py tests/test_pc_dataset.py tests/test_utonia_features.py tests/test_pc_trajectory_model.py tests/test_pc_pipeline.py tests/test_train_pc.py`

Expected: all focused tests pass without an installed GPU/Utonia runtime.

- [ ] **Step 6: Commit.**

```bash
git add configs/train/config_pc_utonia_overfit.yaml train_pc.py tests/test_train_pc.py README.md
git commit -m "feat: add Utonia-conditioned PC overfit training"
```

### 6. Verify repository integration and provide the manual GPU check

**Files:**
- Verify only; do not alter unrelated dirty worktree files.

- [ ] **Step 1: Run formatting/type checks already used by this repository, if configured.** Inspect the project scripts and execute the narrow applicable command(s); do not introduce a new formatter/toolchain.

- [ ] **Step 2: Run the full suite.**

Run: `pytest -q`

Expected: repository test suite passes. If a pre-existing failure occurs, record it separately and do not mask it.

- [ ] **Step 3: Review the diff and status.**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; only the intended implementation commits plus the user's pre-existing dirty files remain.

- [ ] **Step 4: Provide the manual single-GPU smoke command.** Report the exact `accelerate launch --config_file ... train_pc.py --config configs/train/config_pc_utonia_overfit.yaml` invocation after confirming the existing repository launch convention. State expected first-run cache build/download behavior and expected recurring-run cache reuse. Do not claim this succeeds unless it is actually run in a CUDA/Utonia-ready environment.
