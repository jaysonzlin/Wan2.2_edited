# 832×480 I2V Training Variant Design

## Goal

Provide an isolated Wan2.2 TI2V training variant that performs the same training as `train_i2v.py` on native `832×480` Kubric RGBA sequences.

## Architecture

The existing `1280×704` workflow remains unchanged. A new standalone entry point, `train_i2v_832x480.py`, is a near-exact copy of `train_i2v.py`; it loads the same model, encoders, objective, checkpointing, logging, and visualization path, but validates training samples at width 832 and height 480.

The variant uses `configs/train/overfit_kubric_i2v_832x480.yaml`, whose default dataset root is `training_dataset_832x480` and whose data dimensions are 832 by 480. The data directory must contain `sample_*/rgba_00000.png` through `rgba_00048.png`, with each image an RGBA PNG at exactly 832 by 480 pixels.

`submit_832x480.sh` is cloned from `submit_lingbot.sh`. It invokes the new entry point and configuration while retaining Lingbot's training overrides: 10,000 steps, constant learning rate schedule, learning rate 1e-5, batch size 1, gradient norm 2.0, weight decay 0.1, Adam betas 0.9 and 0.95, and the existing checkpoint/visualization cadences. Its job, log, and output identifiers are distinct, with checkpoints written beneath `outputs/i2v_832x480`.

## Runtime Behavior

The TI2V VAE uses a `(4, 16, 16)` stride, so 49 frames at `832×480` produce `13×30×52` spatial-temporal latents. The DiT's `(1, 2, 2)` patch embedding creates a `13×15×26` token grid (5,070 tokens). The existing model derives this grid from runtime tensors and applies three-axis RoPE, so it requires no architecture or sampler modifications.

## Error Handling

The new variant fails during dataset discovery if the root is absent, a sample is incomplete, a frame is not RGBA, or a frame is not exactly `832×480`. The base trainer continues to enforce its existing `1280×704` validation independently.

## Testing

Add tests that establish the new script is a separate runnable Accelerate entry point, that its dataset construction explicitly requests `(832, 480)`, and that the Slurm launcher uses its dedicated script, config, and output path while preserving the chosen Lingbot overrides. Existing tests for `train_i2v.py` and the base dataset must continue to pass.

## Scope

No Kubric rendering, resizing/conversion workflow, model checkpoint change, inference CLI size-list change, or refactor of the original trainer is included.
