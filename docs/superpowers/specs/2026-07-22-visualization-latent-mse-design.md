# Sampled denoised-latent MSE

## Purpose

Log a deterministic, latent-space quality metric every 10 completed optimizer
steps without increasing the MP4 visualization cadence.  The metric compares a
fully sampled I2V latent with the current training batch's matching clean
latent, excluding the conditioned input frame.

## Design

Add `training.denoised_latent_mse_every_steps`, defaulting to `10`, and reject
non-positive or non-integer values during configuration validation.

The visualization sampler will separate latent sampling from MP4 decoding and
writing.  Sampling will continue to use the existing first training condition,
fixed seed, prompt and unconditional prompt, 50 solver steps, time shift, and
CFG scale.  It returns the final latent whether or not a video is requested.

At each completed optimizer step divisible by
`denoised_latent_mse_every_steps`, the trainer will sample from the current
batch's first conditioning frame and calculate `denoised_latent_mse` against
that batch's first clean latent.  It will log the local-process scalar as
`train/denoised_latent_mse`, following the existing logging convention.  The
metric excludes latent time index `0`, which is the clean image condition.

At steps also divisible by `visualization_every_steps` (currently `250`), the
same sampled latent will be decoded and written as the MP4 visualization.  No
additional sample is taken and no separate visualization-only metric is
logged.  At ordinary 10-step metric events, no video is decoded or written.

## Scope and verification

The flow-matching loss, optimizer, scheduler, seed, prompts, sampling
algorithm, MP4 naming, and visualization cadence remain unchanged.  Tests will
cover the positive cadence setting, the sampled-latent reconstruction metric,
sampling without video output, and the source-level cadence arrangement.
