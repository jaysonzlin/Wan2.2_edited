# Three-Object Joint Wan–PhysCtrl Submission Design

## Goal

Add a dedicated Slurm submission script for the cluster-local `td_832x480_3` dataset without altering the existing joint-training job.

## Design

Create `submit_joint_3.sh` as a copy of `submit_joint.sh`. Keep its partition, GPU, CPU, memory, time limit, container invocation, Accelerate configuration, checkpoint-resume setting, and 10,000-step override unchanged.

Change only the job-specific identifiers and training locations:

- Set the Slurm job name and log filename prefix to `wan_joint_physctrl_3`.
- Pass `data.dataset_root=td_832x480_3`, resolved from the existing cluster `PROJECT_DIR` after the script changes directory.
- Pass `logging.output_dir=outputs/joint_wan_physctrl_3` so checkpoints, visualizations, and tracker artifacts cannot overlap the original job.

## Verification

Run Bash syntax validation and inspect the three required values in the new script. The original `submit_joint.sh` remains unchanged.
