# Frozen-Auxiliary Video-Only SimGen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated 4,000-step video-only joint SimGen run that trains Wan while retaining freshly initialized, frozen PC and bridge modules as stable auxiliary computation.

**Architecture:** Extend the existing video-only entry point with an explicit frozen-auxiliary mode. It freezes and verifies the PC/bridge modules, uses an optimizer over Wan alone, and splits reproducible video and PC random streams. Extend the shared joint sampler with an opt-in fixed-PC-state mode so this experiment never feeds a frozen decoder output into a later video denoising step.

**Tech Stack:** Python, PyTorch, Accelerate, Diffusers schedulers, YAML, Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-simgen-video-only-frozen-auxiliary-design.md`

## Global Constraints

- Retain the existing native-RGB, 49-frame, 480×480, three-slot video-only data route and empty text context.
- Start Wan from `Wan2.2-TI2V-5B`; create fresh random PC and bridge modules; never load pretrained PC weights.
- Freeze every PC and bridge parameter before optimizer construction; optimize Wan only.
- Root all deterministic video and PC randomness in `training.seed: 42`, but use independent streams.
- In frozen-auxiliary sampling, retain one synthetic PC history/features/future state for all 50 video steps and do not call the PC decoder or PC scheduler.
- Use a new output lineage, `resume_from_checkpoint: null`, 4,000 steps, checkpoints every 1,000 steps, and sample-0 renders every 1,000 steps.
- Render `sample_490`–`sample_499` in a separate one-GPU job from `checkpoint-4000`.

## Review Focus

- A frozen-auxiliary run must not load a prior trainable-auxiliary checkpoint.
- Normal joint sampling must still call the PC decoder and PC scheduler once per solver step.
- Frozen sampling must preserve the four clean video-history latents through every solver step.
- Frozen modules must be absent from every optimizer group and must fail fast if a gradient appears.
- The renderer must select frozen sampling from the saved run config, not launcher naming.

---

### Task 1: Encode and validate the isolated experiment contract

**Files:**
- Create: `configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml`
- Modify: `training/simgen_video_only_config.py`
- Modify: `tests/test_simgen_video_only_config.py`

**Interfaces:**
- Produces: a config with `training.freeze_pc_and_bridges: true` and `sampling.freeze_auxiliary_pc_state: true` accepted by `load_simgen_video_only_config()`.
- Consumes: the existing strict video-only config loader.

- [ ] **Step 1: Write the failing config tests**

Assert the new YAML has isolated output/run identity, 4,000 steps, null resume, 1,000-step checkpoint/validation/visualization cadence, and both frozen-mode flags. Assert frozen mode with `resume_from_checkpoint: latest` raises `ValueError`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_simgen_video_only_config.py -q`

Expected: FAIL because the new config and frozen-mode validation do not exist.

- [ ] **Step 3: Implement the config contract**

Copy the three-slot YAML, change only experiment identity/schedule fields, and update `validate_simgen_video_only_config(config: dict) -> None` to require the frozen booleans together, null resume, and 4,000 steps for that exact lineage without changing existing lineages.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_simgen_video_only_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml training/simgen_video_only_config.py tests/test_simgen_video_only_config.py
git commit -m "feat: add frozen-auxiliary video-only config"
```

### Task 2: Add split random streams and frozen-module guards

**Files:**
- Modify: `training/simgen_video_only_conditioning.py`
- Modify: `joint_simgen_video_only.py`
- Modify: `tests/test_simgen_video_only_conditioning.py`
- Modify: `tests/test_joint_simgen_video_only.py`

**Interfaces:**
- Consumes: Task 1's `training.freeze_pc_and_bridges` flag.
- Produces: `make_video_only_generators(device, seed) -> tuple[torch.Generator, torch.Generator]`, `freeze_pc_and_bridges(model) -> None`, and optimizer/gradient assertions.

- [ ] **Step 1: Write failing deterministic-RNG and freeze tests**

Prove seed 42 produces repeatable but separate video and PC draws, PC draws cannot shift video draws, Wan receives video-loss gradients, frozen PC/bridge parameters are absent from optimizer groups, and an injected frozen gradient is rejected.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_simgen_video_only_conditioning.py tests/test_joint_simgen_video_only.py -q`

Expected: FAIL because these interfaces do not exist.

- [ ] **Step 3: Implement split RNG and the Wan-only optimizer path**

Derive named video and PC generators from stable distinct offsets of the root seed. Freeze `model.pc_model` and `model.bridges` before constructing AdamW from only `model.wan_model.parameters()` and `optimizer.video` settings. Route flow batching through the video generator and synthetic conditioning through the PC generator for training and scalar validation. Assert frozen parameters have no gradients after backward.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_simgen_video_only_conditioning.py tests/test_joint_simgen_video_only.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/simgen_video_only_conditioning.py joint_simgen_video_only.py tests/test_simgen_video_only_conditioning.py tests/test_joint_simgen_video_only.py
git commit -m "feat: freeze video-only auxiliary modules"
```

### Task 3: Support fixed auxiliary PC state in the shared sampler

**Files:**
- Modify: `wan/joint_pc_pipeline.py`
- Modify: `joint_simgen_video_only.py`
- Modify: `tests/test_joint_pc_pipeline.py`
- Modify: `tests/test_joint_simgen_video_only.py`

**Interfaces:**
- Consumes: Task 1's `sampling.freeze_auxiliary_pc_state` and Task 2's split generators.
- Produces: `JointWanPhysCtrlPipeline.__call__(..., advance_pc_state: bool = True) -> JointSample`; `False` passes `return_pc_output=False`, uses video timestep for PC frame times, and never initializes/steps `pc_scheduler`.

- [ ] **Step 1: Write failing sampler tests**

For `advance_pc_state=False`, assert every model call receives the same future-PC tensor, `return_pc_output` is false, PC scheduler setup/steps never occur, and clean video history remains pinned. Assert video-only sampling selects it from config with independent video/PC generators.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_pc_pipeline.py tests/test_joint_simgen_video_only.py -q`

Expected: FAIL because the sampler always requests and steps PC output.

- [ ] **Step 3: Implement opt-in non-advancing PC mode**

Add `advance_pc_state: bool = True` while retaining normal behavior by default. When false, initialize one PC state from the PC generator, retain it through all video solver steps, use the video timestep for `frame_times`, skip PC scheduler setup/steps, and request no PC decoder output. Update periodic and standalone video-only render paths to pass the config mode and split generators.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_joint_pc_pipeline.py tests/test_joint_simgen_video_only.py -q`

Expected: PASS, including normal joint sampling tests.

- [ ] **Step 5: Commit**

```bash
git add wan/joint_pc_pipeline.py joint_simgen_video_only.py tests/test_joint_pc_pipeline.py tests/test_joint_simgen_video_only.py
git commit -m "feat: hold frozen auxiliary PC state during sampling"
```

### Task 4: Add isolated H200 training and validation launchers

**Files:**
- Create: `submit_joint_simgen_8gpu_2node_mamba_video_only_3slot_frozen_auxiliary_h200_4k.sh`
- Create: `submit_visualize_joint_simgen_video_only_frozen_auxiliary_validation.sh`
- Modify: `tests/test_submit_joint_simgen_video_only_3slot.py`
- Modify: `tests/test_submit_visualize_joint_simgen_video_only_validation.py`

**Interfaces:**
- Consumes: Task 1's config/output lineage and Task 3's config-selected renderer behavior.
- Produces: an 8-GPU two-node submission without a resume override and a one-GPU validation submission targeting `checkpoint-4000`.

- [ ] **Step 1: Write failing launcher tests**

Invoke each shell launcher with fake module/Python executables. Assert the training launcher uses the frozen config without `training.resume_from_checkpoint=latest`; assert the validation launcher defaults to the frozen run's `checkpoint-4000` and checkpoint-local output directory.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_submit_joint_simgen_video_only_3slot.py tests/test_submit_visualize_joint_simgen_video_only_validation.py -q`

Expected: FAIL because the frozen launchers do not exist.

- [ ] **Step 3: Create the Slurm launchers**

Copy the established H200 topology/environment checks, use unique job/log names and the frozen output lineage, invoke the new config with no resume override, and invoke the existing standalone renderer against the frozen default checkpoint. Continue config discovery from `checkpoint.parent/config.yaml`.

- [ ] **Step 4: Run tests and shell syntax checks**

Run: `pytest tests/test_submit_joint_simgen_video_only_3slot.py tests/test_submit_visualize_joint_simgen_video_only_validation.py -q && bash -n submit_joint_simgen_8gpu_2node_mamba_video_only_3slot_frozen_auxiliary_h200_4k.sh && bash -n submit_visualize_joint_simgen_video_only_frozen_auxiliary_validation.sh`

Expected: PASS and no shell syntax output.

- [ ] **Step 5: Commit**

```bash
git add submit_joint_simgen_8gpu_2node_mamba_video_only_3slot_frozen_auxiliary_h200_4k.sh submit_visualize_joint_simgen_video_only_frozen_auxiliary_validation.sh tests/test_submit_joint_simgen_video_only_3slot.py tests/test_submit_visualize_joint_simgen_video_only_validation.py
git commit -m "feat: add frozen-auxiliary SimGen launchers"
```

### Task 5: Verify complete frozen-auxiliary propagation

**Files:**
- Modify: `tests/test_visualize_joint_simgen_video_only_validation.py`
- Modify: `tests/test_joint_simgen_video_only.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: end-to-end unit coverage that standalone rendering honors checkpoint-config frozen sampling and focused suites pass together.

- [ ] **Step 1: Write the failing renderer-selection test**

Mock the CUDA-dependent renderer boundary and assert a frozen checkpoint config preserves its frozen sampling fields through `visualize_validation()` and retains `checkpoint-4000` output naming.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_visualize_joint_simgen_video_only_validation.py -q`

Expected: FAIL until frozen fields reach the sampling call.

- [ ] **Step 3: Complete missing propagation and metrics wiring**

Ensure periodic sample-0 visualization and standalone validation use the config-controlled split-RNG/fixed-PC path. Ensure logs expose zero auxiliary gradient norms and Wan-only learning rate.

- [ ] **Step 4: Run the focused verification suite**

Run: `pytest tests/test_simgen_video_only_config.py tests/test_simgen_video_only_conditioning.py tests/test_joint_simgen_video_only.py tests/test_joint_pc_pipeline.py tests/test_visualize_joint_simgen_video_only_validation.py tests/test_submit_joint_simgen_video_only_3slot.py tests/test_submit_visualize_joint_simgen_video_only_validation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_simgen_video_only.py tests/test_joint_simgen_video_only.py tests/test_visualize_joint_simgen_video_only_validation.py
git commit -m "test: verify frozen-auxiliary rendering contract"
```

## Post-implementation checks

- Run `git diff --check`.
- Run `bash -n` on both new launchers.
- Confirm the saved run config records frozen flags and null resume before submission.
- Do not submit a Slurm job automatically; hand off the scripts and commands for the user to launch.
