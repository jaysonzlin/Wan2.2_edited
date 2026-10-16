#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_2gpu_diagnostic
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
VIDEO_ONLY_CONFIG="configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml"
ACCELERATE_CONFIG="configs/accelerate/h200_2gpu.yaml"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate was not found in ${MAMBA_ENV_PREFIX}." >&2
    exit 1
fi
if [[ ! -f "${PROJECT_DIR}/${ACCELERATE_CONFIG}" ]]; then
    echo "PROJECT_DIR does not contain the two-GPU Accelerate configuration: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi

export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Host: $(hostname)"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Start time: $(date)"

srun --cpu-bind=cores --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",VIDEO_ONLY_CONFIG="${VIDEO_ONLY_CONFIG}",ACCELERATE_CONFIG="${ACCELERATE_CONFIG}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
        nvidia-smi
        "${MAMBA_ENV_PREFIX}/bin/python" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
        exec "${MAMBA_ENV_PREFIX}/bin/accelerate" launch \
            --config_file "${PROJECT_DIR}/${ACCELERATE_CONFIG}" \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip "${MASTER_ADDR}" \
            --main_process_port "${MASTER_PORT}" \
            joint_simgen_video_only_1gpu_diagnostic.py \
            --config "${VIDEO_ONLY_CONFIG}" \
            --expected-processes 2 \
            --optimizer-steps 2 \
            --frozen-auxiliary
    '
