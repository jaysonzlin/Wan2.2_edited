# I2V CFG-scale sweep

## Purpose

Compare classifier-free guidance (CFG) settings for the overfit Wan TI2V
checkpoint while keeping every other sampling input constant.

## Design

`sweep_i2v_inference.py` retains its `EXPERIMENTS` collection but reduces it
to the single scheduler experiment `(1, 50)`. It adds a separate CFG-scale
collection containing `0`, `0.5`, `0.75`, `1`, `2`, and `5`.

Sampling iterates over the scheduler experiment and then the CFG scale. The
same seed, prompt, condition frame, checkpoint, and scheduler parameters are
used for every scale. Each output name includes the CFG scale so no sample is
overwritten, for example `shift_1_steps_50_cfg_0.5.mp4`.

`--list-experiments` remains a GPU-free preview and prints the six planned
output names. Its existing test is updated to make that list the executable
contract. The scheduler helper test remains unchanged.

## Error handling and scope

The existing missing-checkpoint validation and GPU-only sampling behavior are
unchanged. This change does not add CLI arguments, alter model loading, or
modify the training configuration.
