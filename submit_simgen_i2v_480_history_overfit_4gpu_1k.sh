#!/bin/bash
#SBATCH --job-name=simgen_i2v_480_history_overfit_4gpu_1k
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_i2v_480_history_overfit_4gpu_1k_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_i2v_480_history_overfit_4gpu_1k_%j.err

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
        --config_file configs/accelerate/h200_4gpu.yaml \
        train_i2v_simgen_480_overfit.py \
        --config configs/train/overfit_simgen_i2v_480_history.yaml \
        training.max_train_steps=2000 \
        logging.output_dir=outputs/simgen_i2v_480_history_overfit_4gpu_1k \
        logging.wandb_run_name=simgen-i2v-480-history-overfit-4gpu-1k
