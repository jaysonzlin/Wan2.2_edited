# Frozen-auxiliary video-only SimGen experiment

## Purpose

Run a controlled 4,000-step video-only SimGen experiment that retains the
joint Wan/PC architecture but trains only the pretrained Wan branch. It tests
whether the video-only joint model's qualitative behavior comes from training
the PC and bridge modules rather than merely retaining them as fixed auxiliary
computation.

## Scope

The experiment is a new, isolated 8-GPU / 2-node H200 run. It follows the
three-slot video-only data route: native RGB videos only, four video-history
frames, three synthetic PC slots, empty text context, and video flow loss on
the 45 future video-latent frames.

It does not load pretrained PC weights, read real point clouds, train or render
PC trajectories, or resume any prior video-only run with a different
frozen-auxiliary lineage. The isolated run may resume its own latest complete
checkpoint.

## Model and optimizer contract

1. Construct the same pretrained Wan TI2V model, fresh PC trajectory model,
   and eight joint bridge modules as the existing video-only experiment.
2. Immediately freeze every PC-model and bridge parameter at its fresh random
   initialization. The Wan DiT remains trainable.
3. Create an optimizer containing only Wan parameters. Frozen modules must not
   occur in an optimizer group.
4. Before training, assert that the PC and bridge modules have no trainable
   parameters. During logging, report their gradient norms as zero and fail if
   any frozen parameter receives a gradient.

The VAE and text encoder remain frozen as in the existing experiment.

## Randomness contract

`training.seed` remains the single root seed (42). Derive deterministic,
separate device-local generators for:

- video flow noise/timesteps;
- synthetic PC conditioning.

Use those streams for both training and rendering. This prevents the large PC
random draws from shifting the video-noise sequence, while preserving the same
marginal distributions. Synthetic PC conditioning is freshly drawn for every
training forward pass.

## Sampling contract

Sampling conditions on the clean first four video latent frames. For each
rendered video, draw one synthetic PC history, Utonia-like feature tensor, and
future-PC noise state from the PC generator, then keep all three fixed through
the 50 video solver steps.

The frozen PC decoder must not produce a DDIM update that is fed into the next
video denoising step. The sampler therefore needs a video-only frozen-auxiliary
mode that requests no PC output and leaves the synthetic future-PC state
unchanged. It still passes that fixed state through the frozen PC encoder and
bridges at every Wan denoising step.

Video solver state is initialized from the video generator and continues to
pin its four history frames after every solver update.

## Run identity and schedule

Use a new config, output directory, and W&B run name so `latest` cannot load a
checkpoint from a run where PC or bridge weights were trainable. Set
`resume_from_checkpoint: latest`, `max_train_steps: 4000`, checkpoint every
1,000 steps, and retain the existing optimizer/objective values unless the
frozen contract requires a Wan-only optimizer construction.

Render `sample_0` at steps 1,000, 2,000, 3,000, and 4,000. After training,
launch a separate one-GPU validation renderer against `checkpoint-4000` to
render `sample_490` through `sample_499` inclusively. The renderer uses the
same frozen-auxiliary and split-RNG sampling contract.

## Verification

Tests must prove:

- frozen parameters are absent from the optimizer and do not receive gradients;
- the video loss still updates Wan;
- training uses independent deterministic video and PC RNG streams;
- frozen-auxiliary sampling holds PC conditions/state fixed and does not call
  the PC scheduler or decoder;
- sampling continues to pin video history;
- the new config/launchers have isolated identity, same-lineage latest resume, a 4,000-step
  schedule, and the requested visualization cadence.

Run the focused Python tests and shell syntax checks before submitting any
cluster job.
