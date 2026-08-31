# SimGen PC Pretraining Design

## Goal

Train the point-cloud branch used by `joint_simgen.py` before joint training,
using native SimGen samples and the exact history-conditioned DDPM objective.
The resulting PC-only weight export initializes a fresh joint run without
loading Wan or cross-modal bridge weights.

## Architecture

Add a dedicated `pretrain_simgen_pc.py` entry point and a focused
`training/simgen_pc_pretraining.py` module.

The module owns the SimGen-specific transformation from a collated sample to
independent PC trajectories: it flattens the sample's object dimension and
returns point-cloud futures, four-frame histories, and matching cached Utonia
features. It also owns PC-only DDPM x0 loss construction and validation
aggregation. Its interface hides variable object counts from the training
entry point.

`pretrain_simgen_pc.py` owns Accelerate setup, data loaders, model and
optimizer construction, checkpointing, validation, best-model export, and
sample-0 visualization. It constructs the same `PCTrajectoryModel` contract
as joint training:

- 2,048 point identities;
- 49 total frames, with 4 history and 45 predicted frames;
- 8 PC blocks, width 256, 4 heads;
- history-conditioned DDPM predicting x0;
- Utonia feature width inferred from the existing SimGen cache.

The pretrainer uses `SimGenJointDataset` and the existing read-only
`SimGenUtoniaCache`. It never constructs `UtoniaFeatureExtractor`, Wan, or
Wan-to-PC bridge modules. Cached embeddings are inputs; only PC-model
parameters train.

## Data and Objective

Training uses SimGen samples 0 through 127. Validation uses samples 490
through 499 and does not participate in optimization.

One data-loader item is one native SimGen sample per GPU. Every object within
that item is flattened into an independent PC trajectory in the effective
PC-model batch. This is valid because the PC model has no inter-object
attention. All flattened objects retain their own Utonia features.

For every trajectory, noise only the 45 future frames using the established
DDPM scheduler and pass the unmodified four-frame history to the PC model.
Optimize MSE against the clean future trajectory (the x0 target), matching the
joint PC objective. Validation reports an object-weighted global mean so ranks
with different numbers of objects do not receive equal weighting.

## Run Control and Artifacts

A new 4-GPU, single-node Slurm launcher invokes the pretrainer with a batch
size of one SimGen sample per GPU and a maximum of 200,000 optimizer steps.
It configures `resume_from_checkpoint=latest`.

Every 1,000 steps, save a full Accelerate `checkpoint-<step>/` directory.
These directories include model, optimizer, scheduler, and rank-local state so
an interrupted run resumes correctly. `resume_from_checkpoint=latest` sorts
numeric checkpoint directories newest first, attempts to load each in turn,
logs a failed attempt, and falls back to the next older directory when a load
fails. If none can load, it raises one error naming every attempted directory.
Retain only the two newest checkpoint directories; the best-transfer export
below is separate, is never pruned, and does not count toward this limit.

Every 1,000 steps, evaluate the held-out validation split. If the globally
reduced validation PC loss improves, rank zero writes:

- `best_pc_model.pt`: `accelerator.unwrap_model(model).state_dict()` only;
- `best_pc_model.json`: the best validation loss and global step.

The JSON metadata is restored on resume, so later worse validation results
cannot overwrite the transfer artifact. Writes must be rank-zero-only and
atomic, followed by a process barrier before further training.

`best_pc_model.pt` is the exact format consumed by
`joint_simgen.py` through `training.pretrained_pc_weights`. A fresh joint run
sets that field and leaves `training.resume_from_checkpoint` null. A joint
resume must not be combined with PC initialization.

## Visualization

After each validation pass, rank zero runs the history-conditioned DDIM
sampler on native training sample 0. It uses the same cached Utonia features
and four history frames as pretraining. For every object in sample 0, write a
predicted-versus-ground-truth trajectory comparison MP4 under the pretraining
output directory, named with the current global step. Sampling runs under
`torch.no_grad()` and never affects validation metrics, best-model selection,
or gradients.

## Configuration and Errors

Add a dedicated training YAML and validator. The validator requires the fixed
architecture and objective above, sample ranges 0–127 and 490–499, batch size
one, 200,000 maximum steps, and a 1,000-step validation/checkpoint cadence.
The pretrainer must fail before optimizer construction if the Utonia cache is
missing, malformed, or has a feature width incompatible with the PC model.

## Verification

Tests cover:

- flattening variable-object SimGen samples into independent PC trajectories;
- history-DDPM input, target, and Utonia feature alignment;
- object-weighted distributed validation aggregation;
- replacement and persistence of the best PC export and metadata;
- resume fallback and best-metric restoration;
- sample-0 visualization selection and its no-gradient behavior;
- configuration validation and the 4-GPU launcher contract;
- strict loading of the exported state dict into `joint_simgen.py`.

Existing joint scratch training and its launcher remain unchanged.
