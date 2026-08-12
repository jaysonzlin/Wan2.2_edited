# Utonia-conditioned PC trajectory overfit design

## Goal

Add a dedicated `train_pc.py` experiment that overfits the existing PhysCtrl-style
point-cloud trajectory model to object `000`, conditioned on frozen, dense Utonia
features computed from that object's frame-zero XYZ and RGB point cloud.

## Scope

The experiment is additive. It must leave the coordinate-only PC configuration,
the default PC data path, and both joint Wan--PhysCtrl trainers unchanged.

The initial workload is single GPU and intentionally has no validation split. It
may enumerate `objects/000` below every `sample_*` directory, but the intended
dataset currently contains one scene. Training and periodic sampling use the same
Utonia feature condition.

## Data contract

Each selected object remains a `pc.hdf5` trajectory with:

- `point_cloud`: `float32 [49, 1, 2048, 3]`, with fixed physical point identity
  across time;
- `initial_linear_velocity` and `initial_angular_velocity`: `float32 [1, 3]`;
- `rgb`: `uint8 [2048, 3]`, aligned point-for-point with
  `point_cloud[0, 0]` and restricted to `[0, 255]`.

Normals are never read from the dataset; Utonia receives a zero normal per
point. The dataset fails early for a missing, malformed, wrong-dtype, or
out-of-range RGB array.

The Utonia dataset mode selects `objects/<data.object_id>/pc.hdf5`, with the
overfit configuration setting `data.object_id: "000"`. A missing object
directory is an error.

## Utonia cache

`train_pc.py` builds and reuses a persistent, separate feature cache before
constructing its DataLoader. It loads the official `Pointcept/Utonia` weights
through Utonia's Hugging Face loader, sets the encoder to `eval()`, disables
gradients, and releases it before optimizer training starts.

For every selected object, cache construction:

1. uses only frame-zero XYZ and its required RGB;
2. applies Utonia's deterministic single-object preprocessing:
   `normalize_coord=True`, fixed scale, no stochastic augmentation, grid sample,
   color normalization, and zero normals;
3. runs the encoder-only PTv3;
4. restores the documented multi-scale upcast features and maps them through
   Utonia's `inverse` mapping back to the original 2,048 point identities;
5. writes the resulting dense `float32 [2048, D]` tensor to the cache root.

Every cache entry stores the source identity/fingerprint, Utonia checkpoint
fingerprint, transform version/settings, point count, feature width, and
features. The startup routine validates these fields and automatically rebuilds
missing or stale entries. It never modifies source `pc.hdf5` files.

## Model conditioning

`PCTrajectoryModel` accepts an optional per-point
`utonia_features: float [B, 2048, D]`. When absent, its coordinate-only
behavior and parameter layout remain compatible with current callers.

When present, the same source-cloud Utonia features are broadcast across the
clean source frame and all 48 noisy/generated future frames. Immediately after
the existing `PointEmbed`, the model computes:

```text
coordinate token [B, 49, 2048, 256]
|| LayerNorm(Utonia feature) [B, 49, 2048, D]
-> Linear(256 + D, 256)
-> existing fixed PhysCtrl temporal/point-index position embedding
-> existing transformer blocks and output head
```

The LayerNorm is over the Utonia feature channel dimension. The fusion projection
uses normal PyTorch module initialization, matching the rest of the current
trajectory model; it is not identity- or zero-initialized.

Both `PCDDIMPipeline` and `PCFlowPipeline` accept and forward the optional
condition. Baseline and joint callers continue omitting it.

## Configuration and observability

Create `configs/train/config_pc_utonia_overfit.yaml`. It keeps the active
PhysCtrl architecture and DDPM/x0 objective, sets one GPU / batch size one,
selects object `000`, defines a separate `data.utonia_cache_root`, and enables
the Utonia branch. The baseline `config_pc.yaml` remains unchanged.

The trainer logs denoising x0 MSE. At the existing visualization cadence, it
uses a fixed random seed for DDIM sampling and writes the usual predicted vs.
ground-truth trajectory MP4; the Utonia condition is provided to this sampler.

## Error handling

- reject missing object `000`, malformed RGB, and invalid feature tensors;
- reject cache entries whose metadata, source fingerprint, feature width, or
  point count differs from the active condition;
- require the Utonia experiment's cache root and object id in configuration;
- reject a model feature tensor whose batch size or point count differs from the
  trajectory input.

## Verification

Add focused CPU tests with injected/fake Utonia feature extraction to cover:

- object selection and RGB validation;
- cache creation, reuse, and stale rebuild conditions;
- restored feature shape and fixed original point ordering;
- optional model fusion and error checks;
- Utonia condition forwarding in both DDIM and flow pipelines;
- acceptance of the dedicated config while keeping the baseline config valid.

Run the focused PC tests first and the full repository test suite once after
implementation. A manual GPU run is the final integration check because actual
Utonia inference requires its CUDA dependencies and Hugging Face weights.
