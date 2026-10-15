#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k
#SBATCH --partition=gpu_h200
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204,holygpu8a18103,holygpu8a16503
#SBATCH --switches=1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k_%j.err

set -euo pipefail
module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
VIDEO_ONLY_CONFIG="configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    echo "Build it first: ${PROJECT_DIR}/create_mamba_env.sh --from-sif-lock --recreate" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_DIR}/configs/accelerate/h200_8gpu_2node.yaml" ]]; then
    echo "PROJECT_DIR does not contain the eight-GPU Accelerate configuration: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p "logs/nccl-joint-simgen-video-only-3slot-frozen-auxiliary-h200-4k-${SLURM_JOB_ID}"
"${PYTHON_BIN}" -c '
import sys
from training.simgen_video_only_config import load_simgen_video_only_config
load_simgen_video_only_config(sys.argv[1], [])
' "${VIDEO_ONLY_CONFIG}"
export VIDEO_ONLY_CONFIG
MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-joint-simgen-video-only-3slot-frozen-auxiliary-h200-4k-${SLURM_JOB_ID}/nccl.%h.%p.log"
export OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=^lo,docker NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Environment: ${MAMBA_ENV_PREFIX}"
echo "Config: ${VIDEO_ONLY_CONFIG}"
echo "Frozen auxiliary modules; resume latest; 4000 optimizer steps"
echo "Start time: $(date)"

srun --cpu-bind=cores --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
        nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap --format=csv,noheader
        nvidia-smi topo -m
        ibstat || true
        "${MAMBA_ENV_PREFIX}/bin/python" -c "import torch; import flash_attn; import spconv.pytorch; import torch_scatter; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
    '

srun --cpu-bind=cores --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        exec "${MAMBA_ENV_PREFIX}/bin/accelerate" launch \
            --config_file configs/accelerate/h200_8gpu_2node.yaml \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip "${MASTER_ADDR}" \
            --main_process_port "${MASTER_PORT}" \
            joint_simgen_video_only.py \
            --config "${VIDEO_ONLY_CONFIG}"
    '
