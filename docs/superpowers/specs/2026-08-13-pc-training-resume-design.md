# PC Training Resume Design

## Goal

Make `train_pc.py` honor `resume_from_checkpoint` so the Utonia Slurm launcher
can resume a requeued point-cloud training job.

## Behavior

`resume_from_checkpoint` accepts `null`, `"latest"`, or an explicit checkpoint
path. A checkpoint is an Accelerate state directory named `checkpoint-<step>`.

- `null`: begin from zero, as today.
- `"latest"`: consider numeric checkpoint directories in descending step order.
  Load the first complete checkpoint; if a newer directory is incomplete, log
  the failure and try the next older directory. If there are no checkpoints,
  start from zero. If none can load, raise a diagnostic error naming all tried
  checkpoints.
- explicit path: call `accelerator.load_state(path)` and propagate any error.

After a successful load, parse `<step>` from the directory name and use it as
the global step and progress-bar initial position. Accelerate restores the
prepared model, optimizer, learning-rate scheduler, and its managed state.

## Training Semantics

Resume occurs after `accelerator.prepare(...)`, matching the established I2V
trainers and therefore loading into the wrapped objects. The existing loop
continues from the restored global step. It does not restore an intra-epoch
dataloader position; the current shuffled epoch restarts, which is acceptable
for this single-sample overfit experiment.

## Boundaries

Keep the current checkpoint names, save cadence, Utonia feature cache, model
construction, and output layout unchanged. Add configuration validation so bad
resume values fail before model setup.

## Verification

Unit tests cover latest selection, fallback after a failed load, explicit-path
selection, all-failed diagnostics, and PC config validation. Existing trainer
tests verify the restored step is used to initialize the progress bar.
