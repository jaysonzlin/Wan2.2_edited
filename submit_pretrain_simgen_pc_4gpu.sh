#!/bin/bash
#SBATCH --job-name=pretrain_simgen_pc_4gpu
#SBATCH --partition=gpu_requeue
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_a100-sxm4-80gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/pretrain_simgen_pc_4gpu_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/pretrain_simgen_pc_4gpu_%j.err

set -euo pipefail

PROJECT_DIR="/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

nvidia-smi

export PYTHONUNBUFFERED=1

exec singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    -B /tmp:/dev/shm \
    "${PROJECT_DIR}/cur.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_4gpu.yaml \
        pretrain_simgen_pc.py \
        --config configs/train/pretrain_simgen_pc_480_4gpu.yaml \
        training.resume_from_checkpoint=latest
