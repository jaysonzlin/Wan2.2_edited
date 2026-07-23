# I2V LingBot launcher overrides

## Purpose

Run the LingBot-optimizer I2V training job at a learning rate of `1.0e-6` and
store its artifacts separately from the prior requeue-named output directory.

## Design

Update only `submit_h200_i2v_lingbot_optim_requeue.sh`.

- Add `training.learning_rate=1.0e-6` to the `train_i2v.py` overrides.
- Replace `logging.output_dir=outputs/i2v_lingbot_optim_requeue` with
  `logging.output_dir=outputs/i2v_lingbot_optim`.

The script name, Slurm resources, optimizer overrides, resume behavior, shared
I2V config, and tests remain unchanged.

## Verification

Run Bash syntax validation on the edited launcher. No test changes are needed.
