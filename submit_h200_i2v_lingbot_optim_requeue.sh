#!/bin/bash
#SBATCH --job-name=wan_i2v_lingbot_optim
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_i2v_lingbot_optim_%j.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_i2v_lingbot_optim_%j.err

set -euo pipefail

PROJECT_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "============================================================"

nvidia-smi

export PYTHONUNBUFFERED=1

exec singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    -B /tmp:/dev/shm \
    "${PROJECT_DIR}/current.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_i2v.py \
        --config configs/train/overfit_kubric_i2v.yaml \
        data.prompt="" \
        logging.output_dir=outputs/i2v_lingbot_optim_requeue \
        training.resume_from_checkpoint=latest \
        training.max_train_steps=10000 \
        training.checkpoint_every_steps=250 \
        training.checkpoints_total_limit=3 \
        training.lr_scheduler=constant \
        training.max_grad_norm=2.0 \
        training.weight_decay=0.1 \
        training.adam_beta1=0.9 \
        training.adam_beta2=0.95
