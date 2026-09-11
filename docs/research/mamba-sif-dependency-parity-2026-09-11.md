# Mamba/SIF dependency-parity audit (2026-09-11)

## Verdict

The Mamba script is intentionally aligned with the Python, PyTorch, CUDA-wheel,
and Utonia-extension specifications in `current.def`.  It is therefore a
reasonable candidate for the I2V-overfit launcher, but **installed-package
parity is not yet verified**.  This checkout has neither the built SIF nor an
export (`conda list --explicit`, `pip freeze`, or `pip inspect`) for either
environment.  Both recipes also resolve several dependencies from mutable
indexes without exact versions.

## Source comparison

| Area | Mamba recipe | SIF definition | Assessment |
| --- | --- | --- | --- |
| Python and build tools | Conda Forge `python=3.10`, `pip`, `packaging`, and `ninja`. [source](../../create_mamba_env.sh#L72-L81) | Python 3.10, then Conda Forge `packaging` and `ninja`. [source](../../current.def#L44-L50) | Same intended interpreter/tooling. |
| Core numerical / Torch stack | NumPy `>=1.23.5,<2`, `torch==2.4.1`, and `torchvision==0.19.1`, with the PyTorch CUDA 12.4 wheel index. [source](../../create_mamba_env.sh#L83-L98) | The identical NumPy, Torch, torchvision, and CUDA-wheel-index request. [source](../../current.def#L65-L69) | Same explicit core versions and CUDA wheel channel. |
| Training packages | Same requested diffusers, transformers, tokenizers, Accelerate, HF, xFuser, WandB, image, scientific, and utility packages. [source](../../create_mamba_env.sh#L90-L98) | Same package names and version constraints in the temporary requirements file. [source](../../current.def#L70-L98) | Same direct requirements except OpenCV (below). |
| Utonia CUDA extensions | `spconv-cu124` and `torch-scatter` from the Torch 2.4.0+cu124 PyG find-links page. [source](../../create_mamba_env.sh#L100-L101) | Same commands, find-links page, and unpinned extension names. [source](../../current.def#L100-L102) | Same source selection, but not exact extension artifact versions. |
| Flash Attention | Exact `flash_attn-2.6.3+cu126torch2.4` CPython 3.10 Linux wheel URL. [source](../../create_mamba_env.sh#L102-L103) | Same exact wheel URL. [source](../../current.def#L104-L105) | Same binary.  Its `cu126` build tag alongside the cu124 Torch wheel is a shared property, not a Mamba/SIF difference. |
| User-site isolation | Sets `PYTHONNOUSERSITE=1` and passes `--no-user` to pip. [source](../../create_mamba_env.sh#L13-L17) | Sets `PYTHONNOUSERSITE=1` at runtime. [source](../../current.def#L117-L123) | Both prevent accidental imports from `~/.local`. |

## Differences and compatibility risks

1. **Resolver behavior is different.**  Mamba installs the Python stack in one
   transaction with `--upgrade --upgrade-strategy eager`; the SIF first
   installs Torch, then runs a normal `pip install -r` in a fresh image.
   [Mamba source](../../create_mamba_env.sh#L16-L17) [SIF source](../../current.def#L65-L98)
   Direct constraints will remain aligned, but unpinned direct packages and
   their transitive dependencies can resolve differently over time.  Re-running
   the Mamba script can also change those transitive packages.

2. **OpenCV is the one direct-constraint difference.**  Mamba requires
   `opencv-python>=4.9.0.80,<5`, whereas the SIF specifies only
   `>=4.9.0.80`. [Mamba source](../../create_mamba_env.sh#L90-L90)
   [SIF source](../../current.def#L71-L73) This is a conservative Mamba cap;
   it prevents an OpenCV 5 resolution that a future SIF rebuild could accept.

3. **The container supplies runtime/build facilities that Mamba does not.**
   `current.def` is built from CUDA 12.9.2 *devel*, installs compiler, IB/RDMA,
   GL, and other system libraries, and exports CUDA headers, libraries, and
   compiler paths. [base and OS packages](../../current.def#L1-L34)
   [CUDA build environment](../../current.def#L52-L63)
   [runtime environment](../../current.def#L117-L136) The Mamba script creates
   only a Conda/Python prefix, so direct training relies on compatible CUDA,
   RDMA, compiler, and shared libraries supplied by the H200 host.

4. **`pip check` is necessary but insufficient.**  Mamba runs it after
   installation. [source](../../create_mamba_env.sh#L100-L105) It checks Python
   distribution requirements; it does not compare the prefix to the SIF, pin
   resolved versions, or load CUDA extension binaries.

## Evidence from the live Mamba smoke run

The 46039916 NCCL log shows a CUDA driver version `13030`, NCCL
`2.20.5+cuda12.4`, InfiniBand transport, and GPU Direct RDMA enabled.
[source](../../logs/nccl-smoke-46039916/nccl.holygpu8a06103.3164149.log#L4-L15)
This is strong evidence that the Mamba prefix's Torch/NCCL CUDA 12.4 runtime
works across the selected two H200 nodes.  It is not a training-stack parity
test: the smoke launcher executes only `nccl_smoke.py`, while the target I2V
launcher runs `train_i2v_simgen_480_overfit.py`.
[smoke launcher](../../submit_accelerate_smoke_2gpu_2node_mamba.sh#L75-L80)
[target launcher](../../submit_simgen_i2v_480_history_overfit.sh#L55-L69)
The target imports Accelerate and Transformers, and Utonia imports `spconv`,
`torch_scatter`, and optionally `flash_attn`.
[training imports](../../train_i2v_simgen_480_overfit.py#L235-L258)
[Utonia imports](../../wan/utonia/model.py#L24-L39)

## What would establish parity before switching the training launcher

On the same H200 software stack, record the following from the **actual** SIF
and the existing Mamba prefix, then compare normalized outputs:

```bash
# Run once under singularity exec --nv cur.sif, and once with $MAMBA_ENV_PREFIX/bin/python.
python -m pip check
python -m pip freeze --all | sort
conda list --explicit
python -m pip inspect > pip-inspect.json
python -c 'import torch, torchvision, flash_attn, spconv.pytorch, torch_scatter; print(torch.__version__, torch.version.cuda)'
```

Treat the switch as verified only if the import command succeeds in both
environments and the normalized reports either match or have reviewed,
non-runtime-relevant differences.  This specifically exercises the extension
imports the target path needs, which the NCCL smoke test does not.
