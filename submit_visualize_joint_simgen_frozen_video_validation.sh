#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_frozen_validation
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_frozen_validation_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_frozen_validation_%j.err

set -euo pipefail

module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Export CHECKPOINT_PATH as a checkpoint-<step> directory}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ ! -x "${MAMBA_ENV_PREFIX}/bin/python" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors" ]]; then
    echo "CHECKPOINT_PATH is missing model.safetensors: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

args=(
    --config configs/train/joint_simgen_480_8gpu_pc200k_frozen_video.yaml
    --checkpoint "${CHECKPOINT_PATH}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    args+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR:-${CHECKPOINT_PATH%/*}/$(basename "${CHECKPOINT_PATH}")-validation}"
exec "${MAMBA_ENV_PREFIX}/bin/python" visualize_joint_simgen_frozen_video_validation.py "${args[@]}"
