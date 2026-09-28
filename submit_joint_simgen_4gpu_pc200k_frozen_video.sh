#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=joint_simgen_4gpu_pc200k_frozen_video
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --exclude=holygpu8a12204,holygpu8a18103,holygpu8a16503
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_4gpu_pc200k_frozen_video_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_simgen_4gpu_pc200k_frozen_video_%j.err

set -euo pipefail

module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited}"
DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
STARTING_CHECKPOINT="${STARTING_CHECKPOINT:-}"
FROZEN_OUTPUT_DIR="${PROJECT_DIR}/outputs/joint_simgen_8gpu_pc_bridge_frozen_video"

checkpoint_is_complete() {
    local checkpoint="$1"
    local rank
    [[ -d "${checkpoint}" ]] || return 1
    [[ "$(basename "${checkpoint}")" =~ ^checkpoint-[0-9]+$ ]] || return 1
    for required_file in model.safetensors optimizer.bin scheduler.bin; do
        [[ -f "${checkpoint}/${required_file}" ]] || return 1
    done
    for rank in {0..7}; do
        [[ -f "${checkpoint}/random_states_${rank}.pkl" ]] || return 1
    done
}

latest_complete_checkpoint() {
    local candidate latest="" latest_step=-1 candidate_step
    shopt -s nullglob
    for candidate in "${FROZEN_OUTPUT_DIR}"/checkpoint-*; do
        if checkpoint_is_complete "${candidate}"; then
            candidate_step="${candidate##*-}"
            if ((candidate_step > latest_step)); then
                latest="${candidate}"
                latest_step="${candidate_step}"
            fi
        fi
    done
    shopt -u nullglob
    printf '%s' "${latest}"
}

if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
    echo "The Mamba training environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    echo "Build it first: ${PROJECT_DIR}/create_mamba_env.sh --from-sif-lock --recreate" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/configs/accelerate/h200_4gpu.yaml" ]]; then
    echo "PROJECT_DIR does not contain the four-GPU Accelerate configuration: ${PROJECT_DIR}" >&2
    exit 1
fi

LATEST_FROZEN_CHECKPOINT="$(latest_complete_checkpoint)"
if [[ -n "${LATEST_FROZEN_CHECKPOINT}" ]]; then
    RESUME_FROM_CHECKPOINT="${LATEST_FROZEN_CHECKPOINT}"
    RESUME_DESCRIPTION="latest frozen-video checkpoint (${LATEST_FROZEN_CHECKPOINT})"
else
    if [[ -z "${STARTING_CHECKPOINT}" ]]; then
        echo "STARTING_CHECKPOINT is required until ${FROZEN_OUTPUT_DIR} contains a complete checkpoint" >&2
        exit 1
    fi
    if ! checkpoint_is_complete "${STARTING_CHECKPOINT}"; then
        echo "STARTING_CHECKPOINT must be a complete checkpoint-<step> directory for all eight ranks: ${STARTING_CHECKPOINT}" >&2
        exit 1
    fi
    RESUME_FROM_CHECKPOINT="${STARTING_CHECKPOINT}"
    RESUME_DESCRIPTION="starter checkpoint ${STARTING_CHECKPOINT}"
fi

cd "${PROJECT_DIR}"
mkdir -p "logs/nccl-joint-simgen-mamba-${SLURM_JOB_ID}"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-joint-simgen-mamba-${SLURM_JOB_ID}/nccl.%h.%p.log"
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Environment: ${MAMBA_ENV_PREFIX}"
echo "NCCL logs: ${PROJECT_DIR}/logs/nccl-joint-simgen-mamba-${SLURM_JOB_ID}"
echo "Resume setting: ${RESUME_DESCRIPTION}"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap --format=csv,noheader
nvidia-smi topo -m
ibstat || true
"${PYTHON_BIN}" -c "import torch; import flash_attn; import spconv.pytorch; import torch_scatter; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

exec "${ACCELERATE_BIN}" launch \
    --config_file configs/accelerate/h200_4gpu.yaml \
    joint_simgen_frozen_video.py \
    --config configs/train/joint_simgen_480_8gpu_pc200k_frozen_video.yaml \
    "training.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
