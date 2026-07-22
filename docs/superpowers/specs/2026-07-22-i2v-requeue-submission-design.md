# I2V requeue submission script

## Purpose

Provide a clearly named Slurm submission script for the existing Kubric
image-to-video training entrypoint, isolated from the current generic overfit
run's logs and output directory.

## Design

Create `submit_h200_i2v_requeue.sh` as a copy of
`submit_h200_requeue.sh`. It retains the H200 constraint, requeue partition,
single-GPU allocation, Singularity invocation, Accelerate configuration, and
the `train_i2v.py` command.

The new script will use `wan_i2v` in its Slurm job and log filenames. It will
override `logging.output_dir=outputs/i2v_requeue`, ensuring its checkpoints,
visualization MP4s, tracker artifacts, and resume lookup do not overlap the
existing `outputs` run. It will retain `training.resume_from_checkpoint=latest`;
with no checkpoint in the new directory, the training script starts from step
zero.

## Verification

The script will be checked with `bash -n`. A focused text check will verify the
new filename, `train_i2v.py` entrypoint, isolated output override, and absence
of the point-cloud training entrypoint.
