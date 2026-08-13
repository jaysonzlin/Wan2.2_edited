# PC Checkpoint Retention Design

## Goal

Limit Utonia point-cloud training runs submitted with `submit_utonia.sh` to
10,000 optimizer steps and retain only the two newest Accelerate checkpoints.

## Configuration

Add a top-level `checkpoints_total_limit` field to PC configs. It must be a
positive integer. The existing Utonia config retains its 60,000-step default;
the Slurm launcher overrides only the submitted run with:

```bash
max_train_steps=10000 \
checkpoints_total_limit=2
```

## Checkpoint behavior

After `train_pc.py` saves `checkpoint-<step>`, the main process deletes older
numeric checkpoint directories, retaining the newest `checkpoints_total_limit`
directories. Non-numeric directories are ignored. Pruning happens after a
successful save, so the newest state and one older fallback remain available
for `resume_from_checkpoint=latest`.

## Boundaries

Keep the existing save cadence, checkpoint names, Accelerate save/load format,
and resume fallback behavior unchanged. Do not prune checkpoints at startup or
modify the Utonia feature cache.

## Verification

Tests validate PC config limits, numeric checkpoint pruning, launcher
overrides, and preservation of the two latest checkpoint directories.
