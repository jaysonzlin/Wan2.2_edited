#!/bin/bash
#SBATCH --job-name=nccl_smoke_2gpu_2node_torchrun
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204
#SBATCH --switches=1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/nccl_smoke_2gpu_2node_torchrun_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/nccl_smoke_2gpu_2node_torchrun_%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NCCL_SOCKET_IFNAME=^lo,docker
export GLOO_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SMOKE_INIT_TIMEOUT_SECONDS=120

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL logs: ${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}"

srun \
    --cpu-bind=cores \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}",NCCL_DEBUG="${NCCL_DEBUG}",NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}",NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE}",TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    echo "Slurm topology: ${SLURM_TOPOLOGY_ADDR:-not set} (${SLURM_TOPOLOGY_ADDR_PATTERN:-not set})"
    echo ---Allocated-GPU-identity---
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap --format=csv,noheader || true
    nvidia-smi topo -m
    ls -l /sys/class/infiniband || true
    echo ---Host-InfiniBand-port-state---
    ibstat || true
    echo ---Host-RDMA-security---
    id
    id -Z || true
    cat /proc/self/uid_map || true
    ls -lZ /dev/infiniband/uverbs0 || true
    echo ---Host-RDMA-open-probe---
    python3 rdma_open_probe.py || true

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
        echo ---Container-RDMA-security---
        id
        id -Z || true
        cat /proc/self/uid_map || true
        ls -lZ /dev/infiniband/uverbs0 || true
        echo ---Container-RDMA-open-probe---
        python3 rdma_open_probe.py || true
        ls -l /sys/class/infiniband_verbs || true
        for RDMA_PATH in /sys/class/infiniband/*; do
            [[ -e \${RDMA_PATH} ]] || continue
            RDMA_DEVICE=\$(basename \${RDMA_PATH})
            echo ---ibv_devinfo-\${RDMA_DEVICE}---
            ibv_devinfo -d \${RDMA_DEVICE} || true
        done
        ldconfig -p | grep libibverbs || true

        exec torchrun \
            --nnodes=2 \
            --nproc_per_node=1 \
            --node_rank=\"${SLURM_NODEID}\" \
            --master_addr=\"${MASTER_ADDR}\" \
            --master_port=\"${MASTER_PORT}\" \
            nccl_smoke.py
        "
' bash
