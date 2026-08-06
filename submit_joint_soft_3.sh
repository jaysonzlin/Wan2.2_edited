#!/bin/bash
#SBATCH --job-name=joint_soft_3
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:30:00
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_soft_3_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_soft_3_%j.err

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
    "${PROJECT_DIR}/current.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_joint_wan_physctrl.py \
        --config configs/train/joint_wan_physctrl_832x480.yaml \
        data.dataset_root=td_832x480_3_soft \
        logging.output_dir=outputs/joint_soft_3 \
        training.resume_from_checkpoint=latest \
        training.max_train_steps=10000
