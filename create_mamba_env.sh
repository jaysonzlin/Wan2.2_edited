#!/usr/bin/env bash
# Create the shared Mamba environment used by the two-node Accelerate smoke test.

set -euo pipefail

DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
MAMBA_EXE="${MAMBA_EXE:-mamba}"
RECREATE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [--recreate]

Creates or updates the environment at \$MAMBA_ENV_PREFIX.

Environment variables:
  MAMBA_ENV_PREFIX  Environment prefix (default: ${DEFAULT_MAMBA_ENV_PREFIX})
  MAMBA_EXE         Mamba executable (default: mamba)

--recreate removes the existing prefix before creating it again.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate)
            RECREATE=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v "${MAMBA_EXE}" >/dev/null 2>&1; then
    echo "Mamba executable '${MAMBA_EXE}' was not found. Load it and retry, or set MAMBA_EXE." >&2
    exit 1
fi

if [[ -e "${MAMBA_ENV_PREFIX}" && "${RECREATE}" == true ]]; then
    "${MAMBA_EXE}" env remove --yes --prefix "${MAMBA_ENV_PREFIX}"
fi

if [[ ! -x "${MAMBA_ENV_PREFIX}/bin/python" ]]; then
    "${MAMBA_EXE}" create --yes --prefix "${MAMBA_ENV_PREFIX}" \
        --channel conda-forge python=3.10 pip packaging ninja
else
    "${MAMBA_EXE}" install --yes --prefix "${MAMBA_ENV_PREFIX}" \
        --channel conda-forge python=3.10 pip packaging ninja
fi

PYTHON="${MAMBA_ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

# Keep these versions identical to the Python stack installed by current.def.
"${PYTHON}" -m pip install \
    "numpy>=1.23.5,<2" \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    --extra-index-url https://download.pytorch.org/whl/cu124

"${PYTHON}" -m pip install \
    "opencv-python>=4.9.0.80" \
    "diffusers>=0.31.0" \
    "transformers>=4.49.0,<=4.51.3" \
    "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" \
    tqdm imageio easydict ftfy dashscope imageio-ffmpeg "gradio>=5.0.0" \
    "huggingface_hub[cli]" "xfuser>=0.4.1" wandb "PyYAML>=6.0" \
    "Pillow>=10.0" "h5py>=3.10" "matplotlib>=3.8" scipy timm addict psutil einops

"${PYTHON}" -m pip install --no-cache-dir spconv-cu124 torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
"${PYTHON}" -m pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl

"${PYTHON}" - <<'PY'
import accelerate
import torch

print(f"Python environment: {torch.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"Accelerate: {accelerate.__version__}")
PY
