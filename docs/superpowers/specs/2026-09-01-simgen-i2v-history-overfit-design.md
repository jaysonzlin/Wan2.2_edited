# SimGen Video-Only History Overfit Design

## Goal

Measure steps and wall-clock time required for Wan 2.2 TI2V to memorize the RGB video in SimGen `sample_0`, while conditioning on the same four latent-time history slots used by joint SimGen training.

## Scope

The experiment is video-only. It uses no point clouds, Utonia features, joint objectives, or validation samples. It starts from the Wan 2.2 TI2V-5B weights and trains all DiT parameters; VAE and text encoder remain frozen, matching the existing I2V overfit path.

## Data path

Create a focused dataset class for the SimGen layout. It accepts a `sample_root` such as `../simgen/runs/panda_ball_can/sample_0/view_0`, validates exactly `00000000.png` through `00000048.png`, requires 480x480 RGB PNGs, and returns the normalized 49-frame tensor plus the configured prompt. The one-item dataset is the only source for the training loader and visualization batch, so each update sees `sample_0`.

## Training and sampling semantics

A dedicated `train_i2v_simgen_480_overfit.py` reuses the Wan model loading, optimizer, scheduler, accelerator, logging, and flow-matching objective from `train_i2v_832x480.py`. It passes `history_frames=4` to `make_flow_matching_batch`, which holds the first four clean latent-time slots fixed and excludes them from the loss. The visualization sampler must likewise encode and pin the first four latent-time slots on every denoising step. Training noise and timesteps remain newly sampled for each update; visualization uses a fixed seed, prompt, and conditioning input.

## Configuration and artifacts

A dedicated config inherits the 832x480 overfit hyperparameters except for the native 480x480 input and experiment-specific schedule: 10,000 optimizer steps, checkpoint and visualization every 500 steps, and a rolling two-checkpoint limit. Validation is disabled. Visualizations are written beside the reference target video, allowing direct comparison at each interval. No additional `final_dit` export is written, so total retained weight artifacts never exceed two resumable checkpoints.

Use a unique output directory and W&B run name. The initial invocation loads Wan 2.2 weights. Later Slurm requeues resume the latest checkpoint from this unique directory.

## Submission

Add a one-H200 Slurm submit script patterned after `submit_832x480.sh`, using `configs/accelerate/h200_single_gpu.yaml`, the project container, and the dedicated config. It requests a requeue-capable one-GPU job and invokes the dedicated trainer with `training.resume_from_checkpoint=latest`.

## Verification

Unit tests cover strict native sample discovery/loading and the history-latent contract. Lightweight source/config/submit tests assert the 480x480 dimensions, 10,000 steps, 500-step cadence, two-checkpoint limit, disabled validation, H200 single-GPU launcher, and absence of a final model export. Full GPU training is not part of automated test execution.
