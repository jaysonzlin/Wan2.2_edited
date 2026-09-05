#!/bin/bash
#SBATCH --job-name=simgen_i2v_480_history_first128
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_i2v_480_history_first128_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/simgen_i2v_480_history_first128_%j.err

set -euo pipefail

export PROJECT_DIR="/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited"

cd "${PROJECT_DIR}"
mkdir -p logs

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Start time: $(date)"

srun \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    echo "Node rank: ${SLURM_NODEID}; host: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    nvidia-smi

    echo "--- GPU diagnostics before training ---"
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader || true
    nvidia-smi --query-accounted-apps=pid,process_name,gpu_util,mem_util,max_memory_usage,time \
        --format=csv,noheader || true
    nvidia-smi pmon -c 1 || true
    fuser -v /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 || true

    export PYTHONUNBUFFERED=1
    exec singularity exec --nv \
        -B /n/holylabs \
        -B /net/holy-isilon \
        -B /tmp:/dev/shm \
        -B /dev/infiniband \
        "${PROJECT_DIR}/cur.sif" \
        accelerate launch \
            --config_file configs/accelerate/h200_8gpu_2node.yaml \
            --machine_rank "${SLURM_NODEID}" \
            --main_process_ip "${MASTER_ADDR}" \
            --main_process_port "${MASTER_PORT}" \
            train_i2v_simgen_480_overfit.py \
            --config configs/train/overfit_simgen_i2v_480_history.yaml \
            training.resume_from_checkpoint=latest
'
