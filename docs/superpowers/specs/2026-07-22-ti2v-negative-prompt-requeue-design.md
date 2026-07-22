# TI2V negative-prompt requeue run

## Purpose

Add a separate H200 requeue submission script for text-and-image-to-video
training. It uses a fixed positive prompt and Wan's canonical negative prompt
as the CFG baseline during both CFG-dropout training and visualization sampling.

## Design

Add `training.use_wan_negative_prompt` to the I2V training configuration,
defaulting to `false`. `train_i2v.py` will resolve the unconditional prompt to
`""` when disabled and to `ti2v_5B.sample_neg_prompt` when enabled. This keeps
the established empty-prompt behavior unchanged for all existing runs.

The resolved prompt is used in both places that currently hard-code `""`:

1. The 10% text-dropout branch during training, where selected conditional text
   contexts are replaced by the unconditional-context embedding.
2. `save_visualization`, where it supplies the CFG baseline prediction against
   the positive text prompt.

Create `submit_h200_ti2v_negative_requeue.sh` from the existing H200 I2V
requeue script. It will use the positive prompt `Colorful object falls onto the
light gray ground`, set `training.use_wan_negative_prompt=true`, and isolate
checkpoints, visualizations, and resume state under
`outputs/ti2v_negative_requeue`. Its Slurm job and log names will use
`wan_ti2v`.

## Verification

Focused unit tests will verify the prompt resolver's default and enabled
behavior without loading Wan. A source-level training contract will ensure the
resolved prompt is used for CFG dropout and visualization sampling. The new
submission script will receive Bash syntax validation and a focused content
test for its positive prompt, negative-prompt flag, entrypoint, and output path.
