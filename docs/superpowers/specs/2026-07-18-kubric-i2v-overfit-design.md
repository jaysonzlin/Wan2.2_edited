# Kubric I2V Overfit Design

## Goal

Fine-tune the full Wan2.2-TI2V-5B DiT on the local Kubric RGB sequences so
that it overfits the fixed caption `Objects moving in a Kubric simulator`.
The resulting training workflow must use Hugging Face Accelerate, not PyTorch
Lightning, and must run on a separate single-H200 machine.

## Scope

- Use only `training_dataset/sample_*/rgba_00000.png` through
  `rgba_00048.png`.
- Use frame 0 as the image condition and frames 1--48 as the generation
  target.
- Keep all 16 sequences in training; there is no validation split.
- Train at the source and native TI2V resolution, 1280x704, without data
  augmentation.
- Keep the VAE and UMT5 text encoder frozen. Fine-tune the full Wan DiT.
- Exclude depth, actions, proprioception, point clouds, Lightning, and every
  other X-WAM modality.

## Data Contract

The data loader discovers `sample_*` directories below `training_dataset` and
requires exactly the 49 files `rgba_00000.png` to `rgba_00048.png` in each
one. Each file must be an RGBA 1280x704 PNG. The loader composites RGBA over a
constant black background, drops alpha after compositing, converts RGB to
`[-1, 1]`, and retains the original frame order and resolution.

Wan requires `4n+1` total frames because its VAE has temporal stride 4. The
49-frame clip produces 13 temporal latent positions. Frame 0 is a clean
image/video condition and is excluded from the loss; the remaining 48 frames
are noised and supervised.

## Training Architecture

`train_i2v.py` loads `Wan-AI/Wan2.2-TI2V-5B` from a configurable local
checkpoint directory. It creates the frozen VAE and text encoder from the
existing Wan code and places only the DiT in the optimizer. For every batch it:

1. VAE-encodes the 49-frame RGB sequence without gradients.
2. Text-encodes the fixed caption without gradients.
3. Draws a uniform flow-matching timestep, applies Wan's time shift of 5.0,
   and creates a Gaussian-noised latent interpolation.
4. Restores the clean latent corresponding to frame 0 and masks it from the
   target.
5. Has the DiT predict velocity and minimizes masked mean-squared error over
   the target latent positions.

The first-frame image condition follows Wan TI2V's native image-conditioning
mechanism rather than adding an X-WAM modality. The training script uses only
the native Wan video model APIs and modules.

## Optimizer and Runtime

Use single-process Accelerate on one H200 with bf16, FlashAttention and DiT
gradient checkpointing. Use standard `torch.optim.AdamW`, matching X-WAM,
with learning rate `1e-5` and weight decay `0.01`. Use
`get_cosine_schedule_with_warmup` with 200 warmup optimizer updates and 5,000
total optimizer updates. Clip global gradient norm at 1.0.

The default microbatch is 1 with gradient accumulation 4. This is a deliberate
departure from X-WAM's batch-size-4 multi-GPU configuration to accommodate
full-resolution 49-frame full-DiT fine-tuning on one H200.

## Observability and Outputs

Log loss, learning rate and pre-clip gradient norm to Weights & Biases on every
optimizer update. Do not upload qualitative videos to W&B. Every 500 updates,
generate a fixed 49-frame I2V sample from `sample_0` frame 0 and write it to
`outputs/vis/epoch_####.mp4`.

Save an Accelerate state every 250 updates, retain the three newest states,
and export final Wan-compatible DiT weights. The state includes model,
optimizer, scheduler and random-number-generator state to support exact
resumption.

## Files

- `train_i2v.py`: Accelerate entrypoint and training loop.
- `training/overfit_dataset.py`: strict sample discovery, validation and RGBA
  video loader.
- `training/overfit_config.py`: typed YAML loading and command-line overrides.
- `configs/train/overfit_kubric_i2v.yaml`: approved run defaults.
- `configs/accelerate/h200_single_gpu.yaml`: H200 launcher configuration.
- `environment_finetune.yml`: isolated Python 3.10, CUDA 12.4 training Conda
  environment named `wan2-2-finetune`.
- `tests/test_overfit_dataset.py` and `tests/test_overfit_config.py`: CPU-only
  tests for data contract and configuration behavior.
- `requirements.txt`: clearly commented additions for the training workflow,
  including Weights & Biases.

## Invocation

```bash
accelerate launch --config_file configs/accelerate/h200_single_gpu.yaml \
  train_i2v.py --config configs/train/overfit_kubric_i2v.yaml
```

`WANDB_API_KEY` is supplied in the GPU environment. Dataset, checkpoint, W&B,
output, resume and hyperparameter values remain YAML-configurable with
command-line key-value overrides.
