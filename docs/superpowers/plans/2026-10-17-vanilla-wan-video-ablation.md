# Vanilla Wan Video Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible 100,000-step SimGen ablation that trains and samples only the original Wan video model, with no PC or bridge computation.

**Architecture:** A dedicated runner loads `WanModel` directly and uses a small video-only sampling pipeline that owns only the Flow-UniPC state. It reuses the established RGB dataset, encoder, flow-matching helpers, and scheduler configuration, while a strict configuration validator and launcher give the experiment separate lineage and artifacts.

**Tech Stack:** Python, PyTorch, Accelerate, Diffusers Flow-UniPC scheduler, YAML, Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-vanilla-wan-video-ablation-design.md`

## Global Constraints

- Start from `Wan2.2-TI2V-5B`, never from a joint or resumed checkpoint.
- Use only Wan parameters and the established `optimizer.video` AdamW settings.
- Carry no PC/bridge model, optimizer group, tensor, random stream, loss, metric, or visualization artifact.
- Preserve 49 frames, four history frames, 480×480 RGB data, train samples 0--127, validation samples 490--499, seed 42, 100,000 steps, and the established cadence/schedule.
- Render fixed-seed `sample_0` every 1,000 steps; calculate all-ten validation loss every 1,000 steps; render all ten validation videos after the final checkpoint.
- Do not submit a Slurm job automatically.

## Review Focus

- A joint checkpoint or `resume_from_checkpoint` value must be rejected before model construction.
- Wan's native `forward` must receive only video tensors, token times, text context, and sequence length.
- The sampler must repin the four clean history latent frames after every solver update.
- Evaluation RNG must be recreated at each cadence so validation losses and videos are comparable checkpoint-to-checkpoint.
- The final render must cover every configured validation sample, not merely the first sample.

### Task 1: Define the isolated vanilla-video configuration

**Files:**
- Create: `configs/train/simgen_480_8gpu_vanilla_wan_video.yaml`
- Create: `training/simgen_vanilla_video_config.py`
- Create: `tests/test_simgen_vanilla_video_config.py`

**Interfaces:**
- Produces: `load_simgen_vanilla_video_config(path: str | Path, overrides: list[str]) -> dict`.
- Consumes: shared dotted override support from `training.pc_config._apply_override`.

- [ ] **Step 1: Write the failing configuration tests**

Assert the committed YAML uses base Wan initialization, video-only optimizer settings, 100,000 steps, no resume, the fixed splits, and every 1,000-step cadence. Assert the loader rejects PC/bridge-specific keys, non-video optimizer groups, non-null resume, and altered split or cadence values.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q tests/test_simgen_vanilla_video_config.py`

Expected: FAIL because the config and loader do not exist.

- [ ] **Step 3: Implement the strict config contract**

Create the YAML with only video-side fields. Implement the loader and validator with the exact fixed experimental values from the design document, allowing ordinary dotted overrides only where they cannot reintroduce PC state or alter the controlled contract.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest -q tests/test_simgen_vanilla_video_config.py`

Expected: PASS.

### Task 2: Add a native Wan video sampler

**Files:**
- Create: `wan/video_pipeline.py`
- Create: `tests/test_video_pipeline.py`

**Interfaces:**
- Produces: `WanVideoPipeline(model, scheduler, time_shift)` and `WanVideoSample(video_latent: torch.Tensor)`.
- Consumes: `WanModel.forward(x, t, context, seq_len, y=None)` and `expand_latent_timesteps`.

- [ ] **Step 1: Write failing pipeline tests**

Use a lightweight recording video model and scheduler to assert the pipeline passes no PC arguments, makes one native model call per solver timestep, pins the history prefix after every update, and produces identical output for the same generator seed.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q tests/test_video_pipeline.py`

Expected: FAIL because the video-only pipeline does not exist.

- [ ] **Step 3: Implement `WanVideoPipeline.__call__`**

Implement a video-only Flow-UniPC loop accepting the video history latent, full latent shape, empty-text context, inference-step count, and generator. Allocate and update only video latent state; call Wan directly and return `WanVideoSample`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest -q tests/test_video_pipeline.py`

Expected: PASS.

### Task 3: Build the dedicated training and visualization runner

**Files:**
- Create: `simgen_vanilla_video.py`
- Create: `tests/test_simgen_vanilla_video.py`

**Interfaces:**
- Consumes: Task 1's config loader, Task 2's `WanVideoPipeline`, `SimGenVideoDataset`, and existing encoder/flow helpers.
- Produces: `run_training(config: dict) -> None`, a video-only loss helper, periodic sample-0 rendering, and final all-validation rendering.

- [ ] **Step 1: Write failing runner tests**

Mock encoders, flow batching, and a recording Wan model. Assert the loss helper invokes native Wan with only video arguments and returns masked video loss. Assert optimizer construction includes exactly `wan_model.parameters()`. Assert periodic evaluation recreates deterministic validation randomness and final rendering iterates `sample_490` through `sample_499` with distinct reproducible seeds.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q tests/test_simgen_vanilla_video.py`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Load `WanModel` via `load_trainable_dit`, create AdamW from the video group only, and retain standard Accelerate checkpointing. Reuse `SimGenVideoDataset` and the four-history-frame flow objective. At each 1,000-step checkpoint, log the latest training loss plus mean all-ten validation loss and render fixed-seed `sample_0`. After saving `final_checkpoint`, render all ten validation predictions and targets from the live Wan model on rank zero.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest -q tests/test_simgen_vanilla_video.py`

Expected: PASS.

### Task 4: Add the isolated 8-GPU launcher and documentation

**Files:**
- Create: `submit_simgen_8gpu_2node_mamba_vanilla_wan_video.sh`
- Modify: `README.md`
- Create: `tests/test_submit_simgen_vanilla_wan_video.py`

**Interfaces:**
- Consumes: Task 1's config and Task 3's `simgen_vanilla_video.py` entry point.
- Produces: a two-node/eight-H200 Slurm submission command that starts a fresh output lineage.

- [ ] **Step 1: Write the failing launcher test**

Run the launcher against fake Slurm/module/Python executables. Assert it preflights the vanilla config, preserves the established two-node eight-GPU Accelerate topology, invokes `simgen_vanilla_video.py`, uses the new config/output lineage, and passes no resume override.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest -q tests/test_submit_simgen_vanilla_wan_video.py`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher and README entry**

Copy only the established cluster setup, transport checks, environment setup, and accelerate launch pattern. Give the job, log directory, output directory, and run name a `vanilla_wan_video` identity. Document the single `sbatch` command and state that the runner writes the final all-ten validation videos automatically.

- [ ] **Step 4: Run focused verification and shell syntax check**

Run: `pytest -q tests/test_submit_simgen_vanilla_wan_video.py && bash -n submit_simgen_8gpu_2node_mamba_vanilla_wan_video.sh`

Expected: PASS with no shell syntax output.

## Final Verification

- [ ] Run `pytest -q tests/test_simgen_vanilla_video_config.py tests/test_video_pipeline.py tests/test_simgen_vanilla_video.py tests/test_submit_simgen_vanilla_wan_video.py`.
- [ ] Run `git diff --check` and `bash -n submit_simgen_8gpu_2node_mamba_vanilla_wan_video.sh`.
- [ ] Run the repository test suite once; report any pre-existing failures by name.
- [ ] Review the diff against the design document, then commit only the ablation's files without staging unrelated workspace changes.
