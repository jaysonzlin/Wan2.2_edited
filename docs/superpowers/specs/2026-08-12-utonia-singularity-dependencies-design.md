# Utonia dependencies for the Singularity training image

## Goal

Make `current.def` provide the Utonia runtime required by the
Utonia-conditioned point-cloud experiment while retaining its CUDA 12.9 base
image and PyTorch 2.5.1 cu126 stack.

## Build-time package source

The definition copies the sibling `../Utonia` checkout to `/opt/Utonia` with a
`%files` entry and installs it editable with the environment's pip. The image
therefore contains the exact local source used for the build; it does not clone
Utonia from GitHub during `%post`.

## CUDA extensions

The build installs Utonia's required CUDA extensions compatible with its
existing wheel stack: `spconv-cu126==2.3.8` and `torch-scatter` from PyG's
`torch-2.5.1+cu126` index. It adds Utonia's Python dependencies `timm` and
`addict`.

Replace the current prebuilt FlashAttention wheel labelled for Torch 2.4 with
`flash-attn==2.6.3 --no-build-isolation`; it compiles against the image's Torch
2.5.1/cu126 installation and CUDA development toolkit. The build keeps ninja,
packaging, CUDA_HOME, and architecture variables already present.

## Weights

No model weights are baked into the image. The first training/cache-preparation
run continues to call Utonia's Hugging Face loader for `Pointcept/Utonia` and
stores `utonia.pth` beneath
`outputs/utonia_feature_cache/_utonia_checkpoint`. A subsequent cache build
reuses that checkpoint if the mounted/persistent cache root remains available.

## Verification

The definition adds a build-time Python import check for `utonia`,
`spconv.pytorch`, `torch_scatter`, and `flash_attn`, and prints Torch/CUDA
versions. A GPU-enabled container run is still required to verify loading the
actual Utonia checkpoint and extracting features.
