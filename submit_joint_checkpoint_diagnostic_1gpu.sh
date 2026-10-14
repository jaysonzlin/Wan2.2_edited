#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_checkpoint_diagnostic
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_checkpoint_diagnostic_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_checkpoint_diagnostic_%j.err

set -euo pipefail

module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Mamba Python: ${PYTHON_BIN}" >&2
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
"${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
"${PYTHON_BIN}" -m pytest \
    tests/test_joint_bridge.py::test_history_utonia_paired_checkpointing_supports_backward \
    -vv -s
