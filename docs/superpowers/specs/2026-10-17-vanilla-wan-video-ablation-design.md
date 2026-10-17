# Vanilla Wan Video Ablation Design

## Goal

Measure the effect of video--point-cloud bidirectional cross-attention by
training the original Wan video path with no point-cloud or bridge modules in
the forward pass. This is a new isolated run, not a resume or mutation of an
existing joint checkpoint.

## Experimental Contract

- Initialize Wan from the same base `Wan2.2-TI2V-5B` checkpoint as the prior
  joint video-only runs; do not resume any joint run.
- Optimize Wan parameters only, using the existing video optimizer settings.
- Do not instantiate, load, optimize, run, or sample PC trajectory blocks or
  bidirectional bridge blocks.
- Preserve the previous train/validation split (`sample_0`--`sample_127` and
  `sample_490`--`sample_499`), batch size, seed, 49-frame four-history-frame
  video objective, optimizer schedule, and 100,000-step budget.
- Keep the native Wan block forward intact. In particular, the final eight
  blocks execute Wan's ordinary full block implementation rather than the
  joint runner's split block sequence.
- A direct numerical-equivalence test against another Wan invocation is
  intentionally out of scope.

## Training and Evaluation

Training uses only the existing masked flow-matching video loss. At each
1,000-step cadence it records the latest optimization-batch `train/video_loss`
and computes `validation/video_loss` and `validation/loss` across all ten
held-out samples with a fresh deterministic validation RNG stream.

At the same cadence, the runner renders `sample_0` with a fixed inference
seed, writing its prediction and target video. After saving the final
checkpoint, the main process renders prediction and target videos for all ten
validation samples. Every render has deterministic per-sample video noise;
no PC artifacts or PC metrics are created.

## Boundaries

The existing joint video-only and frozen-auxiliary runners retain their current
behavior. The ablation receives a dedicated configuration, entry point, video
sampler, and Slurm launcher so its artifacts and checkpoints cannot be
mistaken for a joint run.

## Success Criteria

The dedicated run can be launched from a clean base Wan checkpoint, its model
and optimizer contain only Wan parameters, its forward and sampling calls have
no PC/bridge inputs, it produces the agreed losses and videos on schedule, and
the final output contains all ten validation visualizations.
