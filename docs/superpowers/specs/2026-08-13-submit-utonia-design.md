# Utonia Slurm Submission Design

## Goal

Provide `submit_utonia.sh` to launch the Utonia-conditioned point-cloud
overfit experiment on a requeueable A100 allocation.

## Script behavior

The script follows `submit_joint_trajectory_12_37.sh` for the cluster
contract: one A100 GPU, `gpu_requeue`, four CPUs, 64 GB memory, a 10.5-hour
limit, requeue enabled, append-mode logs, and the existing Singularity bind
mounts.

It runs `current.sif` from the fixed project directory and launches:

```bash
accelerate launch \
    --config_file configs/accelerate/h200_single_gpu.yaml \
    train_pc.py \
    --config configs/train/config_pc_utonia_overfit.yaml \
    training.resume_from_checkpoint=latest
```

The resume override makes requeued jobs resume from the latest checkpoint in
the configured Utonia output directory. Slurm output and error logs use the
`utonia_%j` naming prefix under the project `logs/` directory.
