# PC DDPM Objective Design

## Goal

Add a PhysCtrl-style DDPM `x0` objective and DDIM sampler to Wan's PC training
workflow while retaining the current flow-matching/UniPC route as a selectable
alternative.  The default PC configuration selects DDPM.

## Configuration

`config_pc.yaml` gains an `objective` section:

```yaml
objective:
  type: ddpm
  num_train_timesteps: 1000
  beta_schedule: linear
  time_shift: 5.0
```

`type` accepts `ddpm` and `flow`.  `time_shift` is used only by flow training
and flow sampling.  Existing `sampling.num_inference_steps` remains shared;
`sampling.solver_order` is used only by UniPC flow sampling.

## Shared model

`PCFlowModel` remains the shared Wan factorized backbone and gains an
`objective_type` constructor argument.

- In `flow` mode it preserves current behavior: accepts a displacement-space
  flow state, requires source time zero, embeds `[p0, p0 + x_t]`, and returns a
  displacement-space flow vector.
- In `ddpm` mode it accepts noisy absolute future positions, uses one DDPM
  timestep for every frame including the source frame, embeds `[p0, x_t]`, and
  returns an absolute x0 prediction `p0 + head_offset`.

PointEmbed, factorized blocks, model size, and the two velocity controls are
shared across both modes.

## Training

The trainer branches on `objective.type`.

- `flow` preserves `make_pc_flow_batch`, flow MSE, and
  `FlowUniPCMultistepScheduler`.
- `ddpm` creates `DDPMScheduler(num_train_timesteps=1000,
  beta_schedule="linear", prediction_type="sample", clip_sample=False)`,
  samples an integer timestep per batch item, noises `points_tgt` directly,
  predicts x0 positions, and uses MSE against `points_tgt`.

The optimizer, learning-rate scheduler, checkpointing, progress logging,
visualization cadence, HDF5 contract, and two controls are unchanged.

## Sampling

`PCFlowPipeline` remains unchanged for flow.  A separate
`PCDDIMPipeline` starts from Gaussian absolute-position latents, applies the
same selected model at each DDIM timestep, and returns its final absolute
future positions without adding `p0` after the scheduler loop.

The visualization branch selects the pipeline matching `objective.type`.

## Validation

Tests cover:

- objective configuration validation and default DDPM selection;
- DDPM batch construction using noised absolute future positions and a shared
  timestep sequence;
- model DDPM residual output and the retained flow output contract;
- DDIM pipeline returning the scheduler's absolute result without a second
  `p0` addition;
- trainer selection through importable helpers; and
- the full pytest suite.

## Non-goals

- Porting CogVideoX/PhysCtrl transformer blocks.
- Removing the flow/UniPC option.
- Changing the fixed PC architecture, data contract, controls, or training
  visualization format.
