# Joint Soft 3 Submission Script Design

## Goal

Add a standalone SLURM submission script for the soft three-object joint-training dataset.

## Design

Create `submit_joint_soft_3.sh` by copying `submit_joint_3.sh` and preserve the existing container launch, H200 resource request, configuration file, checkpoint resume behavior, and maximum training steps.

Apply only these substitutions:

- SLURM job name and log/error filename prefix: `joint_soft_3`.
- CPU request: `#SBATCH --cpus-per-task=4`.
- Training dataset override: `data.dataset_root=td_832x480_3_soft`.
- Training output directory: `logging.output_dir=outputs/joint_soft_3`.

The distinct output directory ensures that `training.resume_from_checkpoint=latest` can only resume checkpoints from this soft-data run.

## Verification

Add or update a focused script-content test that checks the new resource setting, dataset root, and output/log names. Run that focused test and the full pytest suite.

