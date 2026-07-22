# Visualization denoised-latent MSE

## Purpose

Log how closely the fully sampled latent used for a periodic I2V visualization
matches that visualization's ground-truth clean latent. This provides a
latent-space quality metric without changing the flow-matching training loss.

## Design

`save_visualization` will retain its existing 50-step sampling and return the
final sampled latent in addition to writing the MP4. The training loop will
provide the first item of its current `clean_latents` batch to that function
when the existing visualization cadence triggers on the main process.

A focused helper will calculate MSE between the sampled and clean latent
tensors after validating that their shapes match. It will exclude latent time
index `0`, because that input-image condition is kept clean and is not denoised
by the model. All future latent frames and all their channels/spatial elements
will contribute equally.

The main process will log the scalar as
`train/visualization_denoised_latent_mse` through the existing
`accelerator.log` call at the same `global_step` as the visualization. The
metric is not computed or logged at ordinary training steps.

## Scope and verification

Sampling seed, scheduler, prompt conditioning, video decoding, MP4 output, and
the flow-matching objective remain unchanged. Unit tests will cover the masked
MSE value, first-slot exclusion, shape validation, and the visualization
function's returned final latent contract without requiring CUDA.
