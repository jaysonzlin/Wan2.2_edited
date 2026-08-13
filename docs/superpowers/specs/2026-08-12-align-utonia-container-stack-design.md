# Align the container Utonia extension stack

## Goal

Make `current.def` use the Torch/CUDA-extension version set that Utonia's own
working definition uses, so `torch-scatter` is installed from a matching
precompiled PyG wheel instead of falling back to a source build.

## Design

Keep the container's `nvidia/cuda:12.9.2-devel-ubuntu22.04` base image and all
unrelated application dependencies. Change the Conda environment's Python
packages to:

- `torch==2.4.1` and `torchvision==0.19.1` from PyTorch's cu124 index;
- `torch-scatter` from PyG's `torch-2.4.0+cu124.html` index; and
- `spconv-cu124`.

Retain the previously working precompiled
`flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl` installation
unchanged. This matches Utonia's existing `utonia.def` package set despite the
newer CUDA development base; CUDA minor-version compatibility is supplied by
the host driver at runtime.

The static definition test pins these exact declarations and rejects the
previous 2.5.1/cu126 Torch-scatter/spconv declarations. The actual container
build remains the integration verification; no model weights are baked into
the image.
