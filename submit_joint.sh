#!/bin/bash
#SBATCH --job-name=wan_joint_physctrl
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_joint_physctrl_%j.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_joint_physctrl_%j.err

set -euo pipefail

PROJECT_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited"

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
    "${PROJECT_DIR}/current.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_joint_wan_physctrl.py \
        --config configs/train/joint_wan_physctrl_832x480.yaml \
        training.resume_from_checkpoint=latest
