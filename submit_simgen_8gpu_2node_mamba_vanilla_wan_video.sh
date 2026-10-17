#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=simgen_vanilla_wan_video
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --switches=1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_vanilla_wan_video_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_vanilla_wan_video_%j.err

set -euo pipefail
module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
CONFIG="configs/train/simgen_480_8gpu_vanilla_wan_video.yaml"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    exit 1
fi
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -c '
import sys
from training.simgen_vanilla_video_config import load_simgen_vanilla_video_config
load_simgen_vanilla_video_config(sys.argv[1], [])
' "${CONFIG}"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi
export WANDB_DIR="${PROJECT_DIR}/wandb" WANDB_CACHE_DIR="${PROJECT_DIR}/wandb/cache"
export MASTER_ADDR MASTER_PORT OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=^lo,docker NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

mkdir -p "logs/nccl-simgen-vanilla-wan-video-${SLURM_JOB_ID}"
echo "Job ID: ${SLURM_JOB_ID}; rendezvous: ${MASTER_ADDR}:${MASTER_PORT}; config: ${CONFIG}"
srun --cpu-bind=cores --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        exec "${MAMBA_ENV_PREFIX}/bin/accelerate" launch \
            --config_file configs/accelerate/h200_8gpu_2node.yaml \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip "${MASTER_ADDR}" \
            --main_process_port "${MASTER_PORT}" \
            simgen_vanilla_video.py --config configs/train/simgen_480_8gpu_vanilla_wan_video.yaml
    '
