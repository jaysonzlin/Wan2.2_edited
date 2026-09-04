# Joint SimGen 4-GPU Design

## Goal

Run `joint_simgen.py` as one four-process, one-node H200 Accelerate job while
preserving the existing single-GPU workflow and Utonia cache preparation path.

## Execution profile

The new profile consists of two explicit YAML files:

- `configs/accelerate/h200_4gpu.yaml` uses `MULTI_GPU`, four processes, one
  machine, bf16, and the existing static rendezvous settings.
- `configs/train/joint_simgen_480_4gpu.yaml` preserves the fixed SimGen data,
  model, objectives, optimizer, and local batch size of one. It changes
  `training.max_train_steps` to 10,000 and defaults to
  `outputs/joint_simgen_4gpu`.

The resulting effective batch is four samples per optimizer update. Learning
rates remain unchanged. The single-GPU Accelerate and training YAMLs remain
unchanged.

## Distributed training behavior

`run_training` will continue to use `Accelerator.prepare` for DDP model
wrapping and training-loader sharding. It keeps Accelerate's default even
training batches: the 490-sample training split is not divisible by four, so
disabling padding would make DDP ranks take different numbers of optimizer
steps. The unprepared validation loader is evaluated only by rank zero with the
unwrapped model, so its loss is the exact mean over all ten validation samples
without padding or DDP collectives. Each rank keeps the configured base seed; `set_seed` is called with
`device_specific=False`, and its local diffusion generator is seeded from the
same configured value.

At every synchronized optimization step, scalar training metrics are reduced
across ranks and only the main process logs them and updates the progress bar.
Validation runs on rank zero only and logs the mean over exactly samples 490
through 499. Rank zero alone writes the resolved config, checkpoint-pruning
results, tracker events, and visualization artifacts. The validation and
visualization loss forwards use the unwrapped model so they do not issue a
one-rank DDP forward.

## Cache behavior

`--prepare-utonia-cache` remains a one-process operation. The command detects
an Accelerate/torch distributed launch from `WORLD_SIZE`; a value greater than
one raises a clear error instructing the user to run the existing single-GPU
cache command before four-GPU training. Normal training only opens the cache
for read access as it does today.

## Checkpoints and resumes

Every process calls Accelerate's `save_state` and `load_state`, preserving the
model, optimizer, scheduler, scaler, and Accelerate-managed state. A four-GPU
job may resume a one-GPU checkpoint and vice versa when those shared state
files are loadable. After loading, the script always resets the ordinary RNGs
and the diffusion generator to the configured shared base seed on every rank.

This is a deliberate cross-world-size contract: model/optimizer/scheduler and
global step continue, while rank-local RNG is reinitialized rather than claimed
to be bitwise-continuous. Same-world-size Accelerate RNG snapshots are not used
because the requested policy is one shared configured seed per rank on every
run.

## Cluster and documentation

`submit_joint_simgen_4gpu.sh` will use the existing Singularity launch pattern
with `gpu_requeue`, `--constraint=h200`, `--gres=gpu:4`, 16 CPUs, 128 GB RAM,
12 hours, and `--requeue`. It launches the two new YAML profiles. The README
will show the required single-GPU cache command followed by the four-GPU
training command and state the cache/resume/RNG behavior.

## Tests and verification

Unit tests will cover distributed cache rejection, same-seed generator setup,
and training metric reduction without requiring a GPU or model weights. Existing
tests will continue to cover the fixed SimGen configuration and cache source
contract. Static YAML and shell checks will verify the launch profiles. A full
runtime test requires the H200 cluster and is documented as the final
operational validation.
