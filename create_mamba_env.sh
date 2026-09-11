#!/usr/bin/env bash
# Create the shared Mamba environment used by the two-node Accelerate smoke test.

set -euo pipefail

DEFAULT_MAMBA_ENV_PREFIX="/n/holylabs/ydu_lab/Lab/jaysonzlin/wan2-2-mamba"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-${DEFAULT_MAMBA_ENV_PREFIX}}"
MAMBA_EXE="${MAMBA_EXE:-mamba}"
MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-${MAMBA_ENV_PREFIX%/*}/.mamba-pkgs}"
RECREATE=false
REFRESH_METADATA=false

# Do not let this environment resolve imports or pip requirements from ~/.local.
# current.def applies the same isolation at container runtime.
export PYTHONNOUSERSITE=1
PIP_NO_USER_ARGS=(--no-user)
PIP_INSTALL_ARGS=("${PIP_NO_USER_ARGS[@]}" --upgrade --upgrade-strategy eager)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--recreate] [--refresh-metadata]

Creates or updates the environment at \$MAMBA_ENV_PREFIX.

Environment variables:
  MAMBA_ENV_PREFIX  Environment prefix (default: ${DEFAULT_MAMBA_ENV_PREFIX})
  MAMBA_EXE         Mamba executable (default: mamba)
  MAMBA_PKGS_DIRS   Writable package cache (default: ${MAMBA_PKGS_DIRS})

--recreate removes the existing prefix before creating it again.
--refresh-metadata removes this package cache's Conda channel index before installing.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate)
            RECREATE=true
            ;;
        --refresh-metadata)
            REFRESH_METADATA=true
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

mkdir -p "${MAMBA_PKGS_DIRS}"
export CONDA_PKGS_DIRS="${MAMBA_PKGS_DIRS}"

if [[ "${REFRESH_METADATA}" == true ]]; then
    "${MAMBA_EXE}" clean --yes --index-cache
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
"${PYTHON}" -m pip install "${PIP_INSTALL_ARGS[@]}" pip setuptools wheel

# Resolve the complete pure-Python stack together. Keeping NumPy's upper bound
# in this transaction prevents later requirements from upgrading it to NumPy 2.
"${PYTHON}" -m pip install \
    "${PIP_INSTALL_ARGS[@]}" \
    "numpy>=1.23.5,<2" \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    "opencv-python>=4.9.0.80,<5" \
    "diffusers>=0.31.0" \
    "transformers>=4.49.0,<=4.51.3" \
    "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" \
    tqdm imageio easydict ftfy dashscope imageio-ffmpeg "gradio>=5.0.0" \
    "huggingface_hub[cli]" "xfuser>=0.4.1" wandb "PyYAML>=6.0" \
    "Pillow>=10.0" "h5py>=3.10" "matplotlib>=3.8" scipy timm addict psutil einops \
    --extra-index-url https://download.pytorch.org/whl/cu124

"${PYTHON}" -m pip install "${PIP_NO_USER_ARGS[@]}" --no-cache-dir spconv-cu124 torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
"${PYTHON}" -m pip install "${PIP_NO_USER_ARGS[@]}" \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl

"${PYTHON}" -m pip check

"${PYTHON}" - <<'PY'
import accelerate
import numpy
import torch

print(f"Python environment: {torch.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"NumPy: {numpy.__version__}")
PY
