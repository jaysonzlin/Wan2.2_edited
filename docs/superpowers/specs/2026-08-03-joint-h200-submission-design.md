# Joint Wan PhysCtrl H200 Submission Script

## Goal

Provide `submit_joint.sh`, a Slurm submission script for an eight-hour,
single-GPU H200 run of `train_joint_wan_physctrl.py`.

## Execution contract

- Use the `gpu_h200` partition with one GPU, eight CPU cores, 64 GB of memory,
  and an `08:00:00` wall-time limit.
- Run from the shared `Wan2.2_edited` project directory inside the existing
  `current.sif` Singularity image, with the same bind mounts used by the
  project’s other H200 jobs.
- Launch with `accelerate` and `configs/accelerate/h200_single_gpu.yaml`.
- Invoke `train_joint_wan_physctrl.py` with
  `configs/train/joint_wan_physctrl_832x480.yaml` and override
  `training.resume_from_checkpoint=latest`.
- Write Slurm stdout and stderr to distinct joint-training log files in the
  existing `logs/` directory.

## Boundaries

The script will not alter the training YAML, change checkpoint retention, or
enable automatic Slurm requeueing.

## Verification

Validate the script with `bash -n` and inspect it for the required Slurm,
launcher, configuration, and resume arguments.
