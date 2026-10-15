#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_video_only_frozen_auxiliary_validation
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204,holygpu8a18103,holygpu8a16503
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_frozen_auxiliary_validation_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_frozen_auxiliary_validation_%j.err

set -euo pipefail
module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
DEFAULT_CHECKPOINT_PATH="${PROJECT_DIR}/outputs/joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k/checkpoint-4000"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_PATH%/*}/$(basename "${CHECKPOINT_PATH}")-validation}"

if [[ ! -x "${MAMBA_ENV_PREFIX}/bin/python" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors" ]]; then
    echo "CHECKPOINT_PATH is missing model.safetensors: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR}"
exec "${MAMBA_ENV_PREFIX}/bin/python" visualize_joint_simgen_video_only_validation.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite
