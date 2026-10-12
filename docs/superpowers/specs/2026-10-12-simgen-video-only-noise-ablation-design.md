# SimGen Video-Only Noise Ablation Design

## Goal

Train the architecture launched by `submit_joint_simgen_8gpu_2node_mamba.sh`
as a strict video-only ablation. The Wan video pathway keeps its existing
four-frame video history and all trainable modules stay present, but the
point-cloud pathway receives only synthetic noise. No sample-specific
point-cloud, object-count, object-metadata, or Utonia feature value may reach
the model.

## Scope and Baseline Parity

The experiment preserves the reference launcher's hardware allocation,
Accelerate configuration, Wan TI2V initialization, optimizer groups, learning
rates, data split, and cadence:

- two nodes with four H200 GPUs each;
- `configs/accelerate/h200_8gpu_2node.yaml`;
- Wan TI2V checkpoint initialization, with randomly initialized PC and bridge
  modules (`training.pretrained_pc_weights: null`);
- all video, bridge, and PC parameter groups remain trainable;
- train samples 0–127 and validate samples 490–499;
- 1,000-step checkpoint, validation, and sample-0 visualization cadence.

The ablation owns a new config, Slurm submit script, output directory, and
Weights & Biases run name. It must never resume a baseline run.

## Video-Only Data Contract

Introduce a dedicated SimGen video dataset. It reads only the 49 RGB frames
from `sample_<id>/view_0`; it does not open `metadata.json`, `objects/`, any
point-cloud HDF5 file, or Utonia cache record. It returns the native video and
sample ID needed for video supervision and artifacts.

The ablation does not derive object count from a source sample. Every forward
uses exactly three synthetic PC slots. With batch size one and 2,048 points,
the synthetic inputs have these shapes:

- history: `[1, 3, 4, 1, 2048, 3]`;
- noisy future state: `[1, 3, 45, 1, 2048, 3]`;
- Utonia features: `[1, 3, 2048, D]`;
- frame times: `[1, 3, 49]`.

`D` is the existing baseline Utonia feature width. Before the first
production submission, read that width once from the established cache,
record the integer in the ablation YAML as `model.utonia_feature_dim`, and
validate it as a positive integer. The training process must not read the
cache after that; the value preserves the baseline Utonia-fusion parameter
shapes without exposing feature contents.

## Noise and Objective

For each training or scalar-validation model call, synthesize history
coordinates, noisy-future coordinates, and Utonia features independently from
standard normal noise using the existing seeded generator. Retain the
established DDPM timestep/frame-time construction for the PC branch, but do
not use source point clouds to construct it.

Continue to instantiate `JointWanPhysCtrlModel`, `PCTrajectoryModel`, and all
eight bidirectional bridge blocks exactly as in the baseline. The PC branch
and bridges remain trainable and receive whatever gradient flows from the
video loss. Do not freeze, remove, or substitute these modules.

Compute only the existing masked video flow-matching loss. Do not compute,
backpropagate, log, validate, or visualize a point-cloud reconstruction loss.
Validation reports `validation/video_loss` (and the equivalent total loss,
which is identical) over the held-out videos. Training metrics retain video,
bridge, and Wan gradient information but omit PC-loss and per-object metrics.

## Sampling and Visualization

At every 1,000-step validation cadence, render sample 0 from the training
split as a video-only artifact. The render uses the real four-frame *video*
latent history and three synthetic slots. It samples synthetic history and
Utonia feature tensors once from the configured seed, then holds them fixed
across the 50 coupled denoising steps; the PC diffusion state is initialized
and evolved by the existing sampler as usual. This keeps the nuisance
condition stable within one deterministic render.

Save the predicted video and its video target. Do not save PC trajectory
comparisons. Standalone held-out validation rendering, if retained, follows
the same fixed-three-slot sampling rule and writes no PC artifacts.

## Configuration and Safety Checks

The ablation config declares an explicit mode (for example,
`objective.point_conditioning: noise_video_only`) and fixed constants:

- `data.synthetic_object_slots: 3`;
- `model.utonia_feature_dim: D`;
- standard-normal synthetic input generation;
- disabled PC loss and trajectory artifacts.

Its validator rejects other slot counts, missing/non-positive feature widths,
source-cache configuration, or an attempt to combine the ablation mode with a
PC loss. The baseline config and its normal joint-training behavior remain
unchanged.

## Verification

Tests must demonstrate that:

- the video-only dataset neither requires nor opens PC/Utonia inputs;
- synthetic inputs have exactly three slots, expected shapes, independent
  normal draws, and valid DDPM frame times;
- training loss is video-only while PC and bridge parameters remain in the
  optimizer and can receive video-loss gradients;
- validation metrics and visualizations contain no PC loss or PC artifact;
- sampler conditioning is stable across denoising steps yet deterministic for
  the configured seed;
- configuration validation rejects incompatible ablation settings; and
- the new launcher preserves the baseline's two-node/eight-GPU and split /
  cadence contract while using an isolated run path.
