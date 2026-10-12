#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_video_only_3slot
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204,holygpu8a18103,holygpu8a16503
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
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_3slot_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_video_only_3slot_%j.err

set -euo pipefail
module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p "logs/nccl-joint-simgen-video-only-3slot-${SLURM_JOB_ID}"
MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR == 1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to resolve an IPv4 rendezvous address for ${MASTER_HOST}" >&2
    exit 1
fi
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-joint-simgen-video-only-3slot-${SLURM_JOB_ID}/nccl.%h.%p.log"
export OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=^lo,docker NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

srun --cpu-bind=cores --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash -lc '
        set -euo pipefail
        "${MAMBA_ENV_PREFIX}/bin/python" -c "import torch; import flash_attn; import spconv.pytorch; import torch_scatter; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4"
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
            --config configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml \
            data.train_start=0 \
            data.train_end=127 \
            data.validation_start=490 \
            data.validation_end=499 \
            validation.every_steps=1000 \
            visualization.every_steps=1000 \
            training.resume_from_checkpoint=latest
    '
