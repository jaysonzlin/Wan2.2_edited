# Utonia history-conditioned PC trajectory design

## Goal

Add an opt-in Utonia experiment that replaces velocity conditioning with four
observed point-cloud frames while preserving the current velocity-conditioned
experiment and its checkpoint-resume behavior.

## Configuration

Create `configs/train/config_pc_utonia_history_overfit.yaml` by copying the
Utonia overfit experiment's data, optimizer, DDPM, sampling, visualization,
and cache settings. Give the new run a distinct output directory and tracker
project. Its first invocation sets `resume_from_checkpoint: null`, but later
runs may set it to `latest` or a checkpoint path using the existing resume
mechanism.

The model configuration adds:

```yaml
conditioning: history
history_frames: 4
```

Missing `conditioning` defaults to `velocity`, preserving all existing
configuration files. History mode requires `history_frames: 4` for this fixed
49-frame experiment. The prediction horizon is derived as
`data.num_frames - history_frames`, so it is 45 rather than a second,
independently configurable value.

## Data and Utonia contract

The dataset keeps the existing fixed, tracked point ordering. In history mode
it returns a `points_history` tensor containing frames 0--3 and a
`points_tgt` tensor containing frames 4--48. Utonia cache generation remains
unchanged: it extracts and persists one dense feature tensor from frame-zero
XYZ/RGB and validates its `[2048, D]` point ordering.

The model broadcasts that frame-zero Utonia tensor across every temporal
point token. Thus the same feature for point `i` is supplied to point `i` in
each observed history frame and each generated future frame.

## Model and objective

Velocity mode retains the existing one observed frame, 48 future frames, and
linear/angular velocity control tokens without parameter or API changes.

History mode constructs the temporal sequence from four clean history frames
followed by 45 noisy future frames:

```text
P[0], P[1], P[2], P[3], noisy P[4:49]
```

It does not instantiate or use velocity control encoders/tokens for that
mode. The temporal transformer remains bidirectional and unmasked.

History-mode DDPM preserves the existing absolute-position x0 convention and
the model's frame-zero output anchoring; it does not switch to a
displacement-only objective. Its diffusion time input is zero for the four
clean observed frames and the sampled DDPM timestep for all 45 future frames.
Velocity mode retains its existing all-49-timestep behavior.

`condition_drop_rate` remains `0.0`; no classifier-free history dropout mode
is added.

## Sampling and visualization

History-aware sampling accepts `points_history` with shape `(B, 4, 1, N, 3)`,
generates 45 frames, and returns or renders the complete 49-frame trajectory
by prepending the four known frames. It forwards the optional cached Utonia
features at every denoising step. The legacy pipelines retain their existing
velocity-mode signatures and behavior.

## Validation and compatibility

Reject unsupported conditioning modes, non-integer or non-four history frame
counts in history mode, and history-mode checkpoints loaded into an
incompatible velocity model (and vice versa) through normal state-dict shape
validation. Keep `resume_from_checkpoint` valid and functional within either
mode. Do not delete velocity configuration fields, model modules, or pipeline
support.

## Verification

CPU tests cover:

- default velocity-mode configuration and behavior;
- history-mode config validation and derived 45-frame horizon;
- the dataset's 4/45 split;
- history model token construction, zero/active timestep split, and Utonia
  reuse across all 49 temporal frames;
- history DDPM sampling and full-trajectory visualization assembly;
- resume selection and loading for a history-mode checkpoint.

Run focused PC tests, then the full pytest suite. A GPU run with an existing
Utonia cache is the manual integration check.
