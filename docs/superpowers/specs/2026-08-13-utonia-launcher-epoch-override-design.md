# Utonia launcher epoch override

## Goal

Make the Utonia Slurm launcher select a 10,000-epoch limit rather than a
10,000-step limit.

## Change

`submit_utonia.sh` will replace its `max_train_steps=10000` command-line
override with `num_train_epochs=10000`. The base configuration continues to
provide `max_train_steps: 60000`, which remains the global optimizer-step
ceiling enforced by `train_pc.py`.

## Verification

Update the focused launcher test to require the epoch override and reject the
former step override, then run that test.
