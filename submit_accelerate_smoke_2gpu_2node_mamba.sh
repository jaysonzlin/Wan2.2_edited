#!/usr/bin/env bash
#SBATCH --job-name=accelerate_smoke_2gpu_2node_mamba
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204
#SBATCH --switches=1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/accelerate_smoke_2gpu_2node_mamba_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/accelerate_smoke_2gpu_2node_mamba_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate was not found in ${MAMBA_ENV_PREFIX}." >&2
    echo "Create it first: MAMBA_ENV_PREFIX=${MAMBA_ENV_PREFIX} ${PROJECT_DIR}/create_mamba_env.sh" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/configs/accelerate/h200_2gpu_2node.yaml" ]]; then
    echo "PROJECT_DIR does not contain the two-node Accelerate configuration: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p "logs/nccl-smoke-${SLURM_JOB_ID}"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}/nccl.%h.%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SMOKE_INIT_TIMEOUT_SECONDS=120
export PYTHONNOUSERSITE=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Environment: ${MAMBA_ENV_PREFIX}"
echo "NCCL logs: ${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}"

srun --cpu-bind=cores --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
        nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap --format=csv,noheader
        nvidia-smi topo -m
        ibstat || true

        exec "${MAMBA_ENV_PREFIX}/bin/accelerate" launch \
            --config_file "${PROJECT_DIR}/configs/accelerate/h200_2gpu_2node.yaml" \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip "${MASTER_ADDR}" \
            --main_process_port "${MASTER_PORT}" \
            "${PROJECT_DIR}/nccl_smoke.py"
    '
