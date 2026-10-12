# SimGen Video-Only Noise Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing joint Wan/PhysCtrl architecture as a strict video-only SimGen ablation with three fixed synthetic PC slots and no source PC or Utonia inputs.

**Architecture:** Add an isolated video-only dataset, synthetic-conditioning module, config validator, and training entry point. The entry point reuses the baseline model, optimizer, checkpointing, video objective, and coupled sampler; it replaces PC-derived tensors with seeded Gaussian tensors and discards PC outputs, losses, and artifacts.

**Tech Stack:** Python, PyTorch, Accelerate, Diffusers, pytest, YAML, Slurm.

**Spec:** `docs/superpowers/specs/2026-10-12-simgen-video-only-noise-ablation-design.md`

## Global Constraints

- Preserve Wan TI2V, the 8-block PC model, eight bridge blocks, and all three trainable optimizer groups.
- Read only `sample_<id>/view_0` frames. Never open metadata, object folders, PC HDF5, or Utonia cache records.
- Use exactly three slots, 2,048 points, four PC history frames, 45 PC future frames, and standard-normal history/future/Utonia tensors.
- Preserve the baseline 0–127/490–499 split, 100,000 steps, two-node/eight-H200 launch, and 1,000-step checkpoint/validation/sample-0 cadence.
- `model.utonia_feature_dim` is a positive static integer measured once from the established cache; no runtime cache lookup is allowed.
- Optimize and validate only masked video flow loss. Do not emit PC-loss metrics or PC trajectory artifacts.
- Do not change baseline `joint_simgen.py`, its config validator, or existing launchers.

## Review Focus

- Missing metadata and objects must not prevent video-only data loading.
- Wrong slot count, feature width, cache field, or PC loss must fail config validation before model construction.
- Every training and scalar-validation forward must receive fresh noise.
- One sample-0 render must keep history/features fixed across its denoising steps and be deterministic for its seed.
- The launcher must not share run/output/log/resume lineage with the baseline.

---

### Task 1: Video-only data and synthetic conditioning

**Files:**
- Create: `training/simgen_video_only_dataset.py`
- Create: `training/simgen_video_only_conditioning.py`
- Create: `tests/test_simgen_video_only_dataset.py`
- Create: `tests/test_simgen_video_only_conditioning.py`

**Interfaces:**
- Produces `SimGenVideoDataset(dataset_root, sample_ids, *, expected_frames=49, expected_size=(480, 480))` and `simgen_video_collate(samples) -> dict[str, object]`.
- Produces `SyntheticPCConditioning(init_pc, noisy_future_state, utonia_features, frame_times)`, `make_training_conditioning(...) -> SyntheticPCConditioning`, and `make_sampling_conditioning(...) -> tuple[torch.Tensor, torch.Tensor]`.

- [ ] **Step 1: Write failing dataset tests**

Create a sample containing only 49 `view_0` PNG files. Assert `SimGenVideoDataset(...)[0]` returns normalized `[49, 3, 480, 480]` video and `sample_0`; assert collation adds only the batch dimension to tensors.

- [ ] **Step 2: Verify the dataset test is red**

Run: `pytest -q tests/test_simgen_video_only_dataset.py`

Expected: FAIL because the dataset module is absent.

- [ ] **Step 3: Implement the video-only dataset**

Reuse only the joint dataset's RGB validation and `[-1, 1]` normalization. Form paths exclusively under `view_0`; do not import HDF5 or Utonia support.

- [ ] **Step 4: Verify the dataset test is green**

Run: `pytest -q tests/test_simgen_video_only_dataset.py`

Expected: PASS.

- [ ] **Step 5: Write failing conditioning tests**

With a seeded CPU generator and small dimensions, assert training shapes `[B, 3, 4, 1, N, 3]`, `[B, 3, 45, 1, N, 3]`, `[B, 3, N, D]`, `[B, 3, 49]`; assert zero history times, future times in `[0, 999]`, and equal-seed reproducibility. Assert sampling returns unbatched `[3, 4, N, 3]` history and `[3, N, D]` features.

- [ ] **Step 6: Verify the conditioning test is red**

Run: `pytest -q tests/test_simgen_video_only_conditioning.py`

Expected: FAIL because the conditioning module is absent.

- [ ] **Step 7: Implement conditioning constructors**

Generate all synthetic values with `torch.randn(..., generator=generator)`. Sample an independent timestep per `(batch, slot)`, expand it over 45 future positions, and prepend four zero history times. Reject non-positive dimensions; preserve supplied device and dtype.

- [ ] **Step 8: Verify and commit Task 1**

Run: `pytest -q tests/test_simgen_video_only_dataset.py tests/test_simgen_video_only_conditioning.py`

Expected: PASS.

```bash
git add training/simgen_video_only_dataset.py training/simgen_video_only_conditioning.py tests/test_simgen_video_only_dataset.py tests/test_simgen_video_only_conditioning.py
git commit -m "feat: add video-only SimGen conditioning"
```

### Task 2: Strict ablation configuration

**Files:**
- Create: `training/simgen_video_only_config.py`
- Create: `configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml`
- Create: `tests/test_simgen_video_only_config.py`

**Interfaces:**
- Consumes Task 1's fixed dimensions.
- Produces `load_simgen_video_only_config(path: str | Path, overrides: list[str]) -> dict`.

- [ ] **Step 1: Measure and record the baseline Utonia width**

On the training filesystem, inspect one established `outputs/simgen_utonia_feature_cache/*.pt` record and write its feature-channel integer into `model.utonia_feature_dim` in the new YAML. Do not add the cache root to the YAML or runtime code.

- [ ] **Step 2: Write failing validator tests**

Construct a valid config with `objective.point_conditioning: noise_video_only`, three slots, positive explicit width, no cache root, baseline optimizer/geometry, fixed split, and 1,000-step cadence. Parameterize rejections for a different slot count, missing/non-positive width, cache root, wrong mode, non-null pretrained PC weights, and requested PC loss.

- [ ] **Step 3: Verify the validator test is red**

Run: `pytest -q tests/test_simgen_video_only_config.py`

Expected: FAIL because the config module is absent.

- [ ] **Step 4: Implement loader and YAML**

Reuse `_apply_override`. Enforce baseline geometry/history/DDPM/flow/optimizer contracts plus the strict video-only requirements, a distinct output directory, and a W&B run name.

- [ ] **Step 5: Verify and commit Task 2**

Run: `pytest -q tests/test_simgen_video_only_config.py`

Expected: PASS.

```bash
git add training/simgen_video_only_config.py configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml tests/test_simgen_video_only_config.py
git commit -m "feat: configure three-slot video-only SimGen ablation"
```

### Task 3: Video-loss-only trainer

**Files:**
- Create: `joint_simgen_video_only.py`
- Create: `tests/test_joint_simgen_video_only.py`

**Interfaces:**
- Consumes Task 1's dataset/conditioning and Task 2's config loader.
- Produces `build_datasets(config)`, `_video_only_losses(...)`, and `run_training(config) -> None`.

- [ ] **Step 1: Write failing trainer tests**

With recording model/VAE/text-encoder/flow seams, assert `_video_only_losses` supplies three-slot synthetic conditioning, calculates only masked video flow loss, and never requests a PC target. Assert `build_datasets` creates only `SimGenVideoDataset` for the configured ranges. Assert model construction uses the explicit config width rather than a batch tensor.

- [ ] **Step 2: Verify the trainer test is red**

Run: `pytest -q tests/test_joint_simgen_video_only.py`

Expected: FAIL because the entry point is absent.

- [ ] **Step 3: Implement isolated training**

Reuse baseline Accelerate, Wan/PC/bridge construction, optimizer, scheduler, checkpoint, resume, and gradient-norm helpers. Instantiate `PCTrajectoryModel(..., utonia_feature_dim=config["model"]["utonia_feature_dim"])`; create fresh conditioning in every training/scalar-validation forward; backpropagate and validate only video loss. Retain all optimizer groups and gradient norms but omit PC loss/per-object metrics.

- [ ] **Step 4: Verify the trainer test is green**

Run: `pytest -q tests/test_joint_simgen_video_only.py`

Expected: PASS.

- [ ] **Step 5: Add a preserved-trainability regression test**

With a small CPU joint-wrapper fixture, backpropagate video loss from synthetic inputs; assert the optimizer owns Wan, bridge, and PC groups, without requiring every PC parameter to have nonzero gradient at zero-initialized bridge gates.

- [ ] **Step 6: Verify and commit Task 3**

Run: `pytest -q tests/test_joint_simgen_video_only.py tests/test_simgen_video_only_dataset.py tests/test_simgen_video_only_conditioning.py tests/test_simgen_video_only_config.py`

Expected: PASS.

```bash
git add joint_simgen_video_only.py tests/test_joint_simgen_video_only.py
git commit -m "feat: train SimGen video-only noise ablation"
```

### Task 4: Sample-0 rendering and isolated Mamba launch

**Files:**
- Modify: `joint_simgen_video_only.py`
- Create: `submit_joint_simgen_8gpu_2node_mamba_video_only_3slot.sh`
- Create: `tests/test_submit_joint_simgen_video_only_3slot.py`
- Modify: `tests/test_joint_simgen_video_only.py`
- Modify: `README.md`

**Interfaces:**
- Consumes Task 3's trainer and Task 1's `make_sampling_conditioning`.
- Produces deterministic sample-0 video artifacts and an isolated submit-ready launcher.

- [ ] **Step 1: Write failing sampler/visualization tests**

Use a recording pipeline to assert sample-0 uses the first training video, passes three noise slots, and holds the same generated history/features through all sampler calls. Assert only predicted/target video paths are written, no PC visualization function is invoked, and equal seeds produce equal sampler inputs.

- [ ] **Step 2: Verify the visualization test is red**

Run: `pytest -q tests/test_joint_simgen_video_only.py`

Expected: FAIL because rendering is absent.

- [ ] **Step 3: Implement video-only sample-0 rendering**

Reuse baseline flow/DDIM schedulers plus `save_visualization` and `_save_simgen_target_video`; draw sampling history/features once and pass them to `JointWanPhysCtrlPipeline`. Do not inspect or save `future_point_clouds`.

- [ ] **Step 4: Write and verify a failing launcher test**

Model the existing Mamba launcher test with fake `module`, `srun`, `accelerate`, networking, and GPU tools. Assert two-node/eight-GPU arguments, the video-only entry point/config, exact split/cadence/resume overrides, and distinct job/log/output lineage.

Run: `pytest -q tests/test_submit_joint_simgen_video_only_3slot.py`

Expected: FAIL because the launcher is absent.

- [ ] **Step 5: Implement launcher and README handoff**

Copy the reference launcher's resource/NCCL setup, changing only job/log names, entry point, config, and output/run identity. Preserve `PROJECT_DIR` and `MAMBA_ENV_PREFIX` overrides. Document the one-time feature-width lookup and submit command.

- [ ] **Step 6: Verify and commit Task 4**

Run: `pytest -q tests/test_joint_simgen_video_only.py tests/test_submit_joint_simgen_video_only_3slot.py`

Expected: PASS.

```bash
git add joint_simgen_video_only.py submit_joint_simgen_8gpu_2node_mamba_video_only_3slot.sh tests/test_joint_simgen_video_only.py tests/test_submit_joint_simgen_video_only_3slot.py README.md
git commit -m "feat: launch video-only SimGen noise ablation"
```

### Task 5: Full regression verification

**Files:** Modify only Task 1–4 files if regression testing exposes a defect.

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`

Expected: PASS with no failures.

- [ ] **Step 2: Verify new Python syntax**

Run: `python -m compileall -q joint_simgen_video_only.py training/simgen_video_only_dataset.py training/simgen_video_only_conditioning.py training/simgen_video_only_config.py`

Expected: exit code 0.

- [ ] **Step 3: Review diff against the spec**

Confirm no baseline trainer/config/launcher behavior changed; no ablation runtime cache field remains; no PC artifacts are emitted; and all run/resume paths are isolated. Commit only any required regression fix with a specific message.

## Plan Self-Review

- **Spec coverage:** Tasks 1–4 cover the video-only data boundary, fixed three-slot noise, explicit width, video-only loss, preserved architecture/optimizer, deterministic sample-0 rendering, and isolated two-node launcher; Task 5 covers compatibility.
- **Step scan:** Every behavior-changing task starts with a failing test and names its output interface, dimensions, and evidence command. The cache-width lookup is the sole external preflight required by the approved spec.
- **Type consistency:** Task 1's batched conditioning is consumed by Task 3; its unbatched sampler conditioning is consumed by Task 4.
- **Review focus:** Each listed likely failure has a test in Tasks 1–4.
- **Proportion:** The plan fixes interfaces and tests without transcribing trainer internals.
