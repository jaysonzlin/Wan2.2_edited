#!/bin/bash
#SBATCH --job-name=history_utonia
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/history_utonia_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/history_utonia_%j.err

set -euo pipefail

PROJECT_DIR="/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
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
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_pc.py \
        --config configs/train/config_pc_utonia_history_overfit.yaml \
        num_train_epochs=10000 \
        checkpoints_total_limit=2 \
        resume_from_checkpoint=latest
