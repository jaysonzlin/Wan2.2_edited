#!/bin/bash
#SBATCH --job-name=nccl_smoke_8gpu_2node
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/nccl_smoke_8gpu_2node_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/nccl_smoke_8gpu_2node_%j.err

set -euo pipefail

PROJECT_DIR="/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited"

cd "${PROJECT_DIR}"
mkdir -p "logs/nccl-smoke-${SLURM_JOB_ID}"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi

# Observe NCCL's automatic network selection; do not force an interface or HCA.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}/nccl.%h.%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL logs: ${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}"

srun \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}",NCCL_DEBUG="${NCCL_DEBUG}",NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}",NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE}",TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    nvidia-smi topo -m
    ls -l /sys/class/infiniband || true

    exec singularity exec --nv \
        -B /n/holylabs \
        -B /net/holy-isilon \
        -B /tmp:/dev/shm \
        -B /dev/infiniband \
        "${PROJECT_DIR}/cur.sif" \
        bash -lc "
        echo ---Container-RDMA-diagnostics---
        ls -l /dev/infiniband || true
        ibv_devices || true
        ls -l /sys/class/infiniband_verbs || true
        for RDMA_PATH in /sys/class/infiniband/*; do
            [[ -e \${RDMA_PATH} ]] || continue
            RDMA_DEVICE=\$(basename \${RDMA_PATH})
            echo ---ibv_devinfo-\${RDMA_DEVICE}---
            ibv_devinfo -d \${RDMA_DEVICE} || true
        done
        ldconfig -p | grep libibverbs || true

        exec accelerate launch \
            --config_file configs/accelerate/h200_8gpu_2node.yaml \
            --machine_rank \"${SLURM_NODEID}\" \
            --main_process_ip \"${MASTER_ADDR}\" \
            --main_process_port \"${MASTER_PORT}\" \
            nccl_smoke.py
        "
' bash
