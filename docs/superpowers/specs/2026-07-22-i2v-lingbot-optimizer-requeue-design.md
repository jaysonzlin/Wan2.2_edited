# I2V LingBot optimizer requeue run

## Purpose

Add a separate H200 requeue launcher for the existing empty-prompt I2V
experiment. It will use LingBot's video-training optimizer values without
changing the optimizer behavior of current I2V runs.

## Design

Extend the `training` section of `configs/train/overfit_kubric_i2v.yaml` with
the configurable AdamW parameters `adam_beta1`, `adam_beta2`, and
`adam_epsilon`. Their defaults will remain PyTorch AdamW's defaults: `0.9`,
`0.999`, and `1e-8`, respectively. `train_i2v.py` will pass all three values,
alongside its existing learning rate and weight decay, to `torch.optim.AdamW`.

This preserves the existing I2V optimizer for all configurations that do not
override the new options.

Create `submit_h200_i2v_lingbot_optim_requeue.sh` by copying
`submit_h200_i2v_requeue.sh`. The new script will retain its empty prompt,
H200 resources, requeue behavior, entrypoint, checkpoint settings, and
constant schedule. It will set these LingBot-aligned optimizer overrides:

- `training.max_grad_norm=2.0`
- `training.weight_decay=0.1`
- `training.adam_beta1=0.9`
- `training.adam_beta2=0.95`

It will use `outputs/i2v_lingbot_optim_requeue` so checkpoints, resume state,
and visualizations do not overlap the standard empty-prompt I2V run. Slurm job
and log identifiers will use `wan_i2v_lingbot_optim`.

## Verification

Add a focused unit test that constructs the I2V AdamW optimizer from a minimal
training dictionary and checks its beta, epsilon, and weight-decay parameter
group values. Add a submission-script test that checks the empty prompt,
LingBot optimizer overrides, output directory, job name, and `train_i2v.py`
entrypoint. Run Bash syntax validation and the full repository pytest suite in
the `das` Conda environment.
