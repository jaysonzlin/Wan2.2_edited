# Wan PC Flow Model Design

## Goal

Add a Wan2.2_edited-native point-cloud trajectory workflow that remakes the
PhysCtrl PC model topology with selected Wan architectural primitives and a
Wan-style shifted continuous flow-matching objective. It trains from the
existing `training_dataset/sample_*/pc.hdf5` files and returns absolute future
point clouds.

## Scope and ownership

All implementation belongs to `Wan2.2_edited`. The existing `edited-physctrl`
repository is a behavioral and architectural reference only; it is not an
imported runtime dependency. Existing TI2V training and inference, including
`WanModel`, VAE/T5 loading, and `train_i2v.py`, remain unchanged.

The workflow has dedicated PC modules, trainer, config, tests, and
visualization. It follows the repository's existing root-entrypoint,
`training/` helper, YAML config, and CPU-test conventions.

### No surplus PhysCtrl port

The remake is defined only by the behavior selected in PhysCtrl's
`config_pc.yaml`. It must be a clean Wan-native implementation, not a copy of
`edited-physctrl/src/model/spacetime.py` or its generic CogVideoX wrapper.

Do not port any unused PhysCtrl alternatives or optional machinery, including:

- spatial-only, temporal-only, v2, and v3 transformer-block variants;
- MDM/MCG models and material, force, drag, gravity, class, floor, or
  coefficient conditioning;
- optional class embeddings, OFS embeddings, LoRA/PEFT processor plumbing,
  generic Diffusers model/config mixins, and checkpoint-format compatibility;
- image/video patching, patch-size branches, learned-position alternatives,
  or generic text-sequence configuration;
- PhysCtrl's DDPM/DDIM pipeline, `pred_offset` head behavior, and
  classifier-free condition-drop path.

Only these configuration-selected PC features transfer: `PointEmbed`, 8
factorized spatial-temporal blocks at width 256 with 4 x 64-dimensional heads,
a prepended source frame, fixed point-index/temporal sinusoidal positions, two
linear velocity tokens, and zero condition drop. The new Wan flow objective,
flow head, per-frame time interface, and flow solver replace the original
DDPM-specific pieces.

## Data contract

The dataset accepts only files at `sample_*/pc.hdf5` under the configured
dataset root. Every file must contain:

- `point_cloud` with exact shape `(49, 1, 2048, 3)`;
- `initial_linear_velocity` with exact shape `(1, 3)`;
- `initial_angular_velocity` with exact shape `(1, 3)`.

Frame zero is the clean initial cloud `p0`; frames one through 48 are the
absolute target clouds `pf`. The dataset rejects missing keys or a nonmatching
shape with a descriptive `ValueError` or `KeyError`.

## Representation and flow objective

The denoised variables are future displacements:

`d = pf - p0`.

For each batch, sample `u` uniformly on `[0, 1]`, apply Wan's rational time
shift with configurable default `s = 5`, then construct:

`t = s * u / (1 + (s - 1) * u)`

`d_t = (1 - t) * d + t * epsilon`, where `epsilon` is Gaussian noise.

The model predicts the flow field `epsilon - d` and the loss is mean squared
error over all future displacement elements. The clean source condition is not
part of the noised/loss-bearing trajectory. Its time is exactly zero; each
future frame receives the shared sampled shifted time, yielding
`[0, t, ..., t]`.

The model head predicts a flow vector directly. It never adds `p0`. The
pipeline integrates the predicted flow to produce final displacement `d_hat`
and returns absolute predictions `p0 + d_hat`.

## Model architecture

The production configuration preserves the relevant PhysCtrl PC settings:

- 2,048 points and 48 future frames;
- 8 transformer blocks;
- hidden width 256;
- 4 attention heads of width 64;
- a clean, prepended source frame;
- Fourier-coordinate `PointEmbed`;
- fixed point-index and temporal sinusoidal embeddings;
- two condition tokens formed by one linear projection of each initial
  velocity vector;
- zero condition-drop rate.

The source frame contains absolute `p0`; future stream inputs contain noised
displacements. The source slot's time-zero embedding identifies this deliberate
condition/data distinction. No additional per-point geometry branch is added.

Each factorized block retains PhysCtrl's operation order:

1. joint spatial attention over the velocity condition tokens and per-frame
   point tokens;
2. a joint modulated GELU MLP that updates both streams;
3. temporal attention over the 49-frame track of each point, without a second
   temporal MLP.

Velocity tokens evolve through the spatial operations exactly as in PhysCtrl;
they are not Wan-style immutable cross-attention context.

The implementation adapts Wan primitives for these PC tensor layouts:

- Q/K RMS normalization;
- time-conditioned modulation and residual gates;
- GELU MLPs;
- a modulated normalized flow-output projection.

It does not reuse Wan's VAE, 3D `Conv3d` patch embedder, unpatchify head, T5
encoder, text cross-attention, flattened video-token attention, or 3D grid
RoPE. Those components assume a dense video lattice or text conditioning and
are incompatible with the fixed point-cloud contract.

## Training, sampling, and visualization

`train_pc.py` is a dedicated Accelerate entry point. It reads the new
`configs/train/config_pc.yaml`, writes its resolved config to the run output,
uses the configured AdamW/cosine-warmup settings, supports existing checkpoint
resume semantics, and trains only the PC flow model.

`PCFlowPipeline` wraps Wan's existing `FlowUniPCMultistepScheduler`. It starts
with Gaussian future displacement frames, calls the model with the clean source
cloud and both velocity conditions at every solver step, and returns the 48
absolute predicted future frames. Classifier-free guidance is not trained or
enabled because `condition_drop_rate` remains zero.

At the configured visualization cadence, the trainer samples a deterministic
one-item batch and writes an MP4 comparing predicted and ground-truth point
cloud trajectories under the run's `vis_dir`.

## Configuration and dependencies

`configs/train/config_pc.yaml` is the source of truth for the new workflow. It
contains the copied PhysCtrl PC optimizer, cadence, capacity, and dataset
values plus flow-specific settings: `prediction_type: flow`, `time_shift: 5.0`,
and PC solver settings. It does not read any config from `edited-physctrl`.

`h5py` becomes an explicit project dependency because the workflow reads the
HDF5 PC dataset directly.

## Validation

CPU tests cover:

- strict HDF5 dataset validation and target/source split;
- shifted flow construction, target flow, and protected time-zero source slot;
- PC model input validation, output shape, and direct flow semantics;
- the pipeline's solver integration contract and final `p0 + d_hat`
  conversion;
- a one-step Accelerate trainer smoke test with tiny injectable PC dimensions;
- visualization output creation from a tiny synthetic trajectory.

The implementation must leave the existing Wan TI2V test suite passing.
