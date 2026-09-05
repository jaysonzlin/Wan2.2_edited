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
export RDMA_PROVIDER_CONFIG_DIR="${RDMA_PROVIDER_CONFIG_DIR:-/etc/libibverbs.d}"

RDMA_LIBRARIES=()
while IFS= read -r RDMA_LIBRARY; do
    RDMA_LIBRARIES+=("${RDMA_LIBRARY}")
done < <(
    ldconfig -p \
        | awk '/libibverbs\.so|libmlx5.*\.so|librdmacm\.so|libnl-3\.so|libnl-route-3\.so/ { print $NF }' \
        | sort -u
)
if (( ${#RDMA_LIBRARIES[@]} == 0 )); then
    echo "Unable to locate host RDMA libraries with ldconfig" >&2
    exit 1
fi

RDMA_PROVIDER_FOUND=false
for RDMA_PROVIDER_CONFIG in "${RDMA_PROVIDER_CONFIG_DIR}"/*mlx5*.driver; do
    [[ -f "${RDMA_PROVIDER_CONFIG}" ]] || continue
    RDMA_PROVIDER_NAME=$(tr -d '\n' < "${RDMA_PROVIDER_CONFIG}")
    RDMA_PROVIDER_PATH=$(ldconfig -p | awk -v name="${RDMA_PROVIDER_NAME}" '$1 == name { print $NF; exit }')
    if [[ -z "${RDMA_PROVIDER_PATH}" ]]; then
        for RDMA_LIBRARY_DIR in /lib64 /usr/lib64; do
            RDMA_PROVIDER_PATH="${RDMA_LIBRARY_DIR}/${RDMA_PROVIDER_NAME}"
            [[ -f "${RDMA_PROVIDER_PATH}" ]] && break
            RDMA_PROVIDER_PATH=""
        done
    fi
    if [[ -z "${RDMA_PROVIDER_PATH}" ]]; then
        echo "Unable to locate RDMA provider ${RDMA_PROVIDER_NAME}" >&2
        exit 1
    fi
    RDMA_LIBRARIES+=("${RDMA_PROVIDER_PATH}")
    RDMA_PROVIDER_FOUND=true
done
if [[ "${RDMA_PROVIDER_FOUND}" != true ]]; then
    echo "Unable to locate an mlx5 RDMA provider configuration" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL logs: ${PROJECT_DIR}/logs/nccl-smoke-${SLURM_JOB_ID}"
printf 'Host RDMA libraries:\n%s\n' "${RDMA_LIBRARIES[@]}"

srun \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}",NCCL_DEBUG="${NCCL_DEBUG}",NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}",NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE}",TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    nvidia-smi topo -m
    ls -l /sys/class/infiniband || true

    RDMA_LIBRARY_BIND_ARGS=()
    for RDMA_LIBRARY in "$@"; do
        RDMA_LIBRARY_NAME=$(basename "${RDMA_LIBRARY}")
        RDMA_LIBRARY_REALPATH=$(realpath "${RDMA_LIBRARY}")
        RDMA_LIBRARY_BIND_ARGS+=(
            -B "${RDMA_LIBRARY_REALPATH}:/tmp/${RDMA_LIBRARY_NAME}"
        )
    done
    RDMA_CONFIG_BIND_ARGS=()
    if [[ -d "${RDMA_PROVIDER_CONFIG_DIR}" ]]; then
        RDMA_CONFIG_BIND_ARGS=(-B "${RDMA_PROVIDER_CONFIG_DIR}:/etc/libibverbs.d")
    fi

    exec singularity exec --nv \
        -B /n/holylabs \
        -B /net/holy-isilon \
        -B /tmp:/dev/shm \
        -B /dev/infiniband \
        "${RDMA_LIBRARY_BIND_ARGS[@]}" \
        "${RDMA_CONFIG_BIND_ARGS[@]}" \
        "${PROJECT_DIR}/cur.sif" \
        bash -lc "
        echo ---Container-RDMA-diagnostics---
        ls -l /dev/infiniband || true
        ibv_devices || true
        ldconfig -p | grep libibverbs || true
        ls -l /tmp/libibverbs.so.1 || true
        ldd /tmp/libibverbs.so.1 || true
        export LD_LIBRARY_PATH=/tmp\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}

        exec accelerate launch \
            --config_file configs/accelerate/h200_8gpu_2node.yaml \
            --machine_rank \"${SLURM_NODEID}\" \
            --main_process_ip \"${MASTER_ADDR}\" \
            --main_process_port \"${MASTER_PORT}\" \
            nccl_smoke.py
        "
' bash "${RDMA_LIBRARIES[@]}"
