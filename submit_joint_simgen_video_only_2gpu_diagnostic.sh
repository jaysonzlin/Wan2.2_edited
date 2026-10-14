#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_2gpu_diagnostic
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_2gpu_diagnostic_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_2gpu_diagnostic_%j.err

set -euo pipefail

module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba}"
TORCHRUN_BIN="${MAMBA_ENV_PREFIX}/bin/torchrun"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "Missing Mamba torchrun: ${TORCHRUN_BIN}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Start time: $(date)"
nvidia-smi
"${MAMBA_ENV_PREFIX}/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
    joint_simgen_video_only_1gpu_diagnostic.py --expected-processes 2
