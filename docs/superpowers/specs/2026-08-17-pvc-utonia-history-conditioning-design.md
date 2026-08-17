# PVC Utonia History Conditioning Design

## Goal

Add a dedicated point-view-conditioned (PVC) training experiment. It retains
the fixed 49-frame, four-frame-history DDPM trajectory task, but conditions
each frame's spatial attention on that frame's RGB-D-derived point view.

The existing `train_pc.py` history and velocity experiments remain unchanged.

## Chosen approach

PVC uses full, bidirectional spatial self-attention over the concatenation of
trajectory and view tokens. This was selected over cross-attention because
the view tokens must be part of the spatial sequence and retain their spatial
states across transformer blocks. It was also selected over adding another
mode to `train_pc.py`, which would make the legacy trajectory training and
PVC checkpoint contracts unnecessarily interdependent.

## Experiment boundary

Create `train_pvc.py` as a dedicated, history-only entry point. It accepts the
same YAML/override pattern as the current trainer but validates the PVC-fixed
contract:

- 49 frames, 2,048 trajectory points, four history frames, and 45 DDPM future
  frames;
- history conditioning, DDPM, and Utonia are required;
- velocity and flow modes are rejected;
- `train_pc.py`, its velocity path, and its existing history checkpoints do
  not change.

Create `configs/train/config_pvc_utonia_history_overfit.yaml` from the current
history config with these distinct artifacts:

```yaml
output_dir: ./outputs/pvc_trajectory_utonia_history_overfit
tracker_project_name: pvc_trajectory_utonia_history_overfit
data:
  dataset_root: td_832x480_3_soft
  point_view_utonia_cache_root: ./outputs/utonia_point_view_feature_cache
```

The relative dataset root intentionally assumes the existing remote launch
directory, where `td_832x480_3_soft` is directly beneath that directory.

## Data and cache contract

For every `sample_*`, PVC reads `point_views/0000.h5` through
`point_views/0048.h5`. Each file must contain finite float XYZ data and RGB
compatible with the existing Utonia preprocessing. A view may have zero to
2,048 points; missing, malformed, non-finite, or oversized views fail during
dataset validation. `depth.h5` is an upstream extraction artifact and is not
read by PVC.

The dataset returns these additional tensors:

```text
point_views:      float32 [49, 2048, 3]
point_view_mask:  bool    [49, 2048]
point_view_utonia_features: float32 [49, 2048, D]
```

Real view points are packed in file order and padded with zeros. The Boolean
mask is the source of truth; zero values do not themselves represent padding.

The point-view Utonia cache has one atomic record per sample,
`point_view_utonia_features.pt`. Its features are `[49, 2048, D]`, its mask is
`[49, 2048]`, and its metadata includes a source fingerprint covering all 49
view HDF5 files plus the existing Utonia checkpoint/preprocessing identity.
Only real XYZ/RGB points are passed to frozen Utonia; their features are then
padded. A changed source file, checkpoint, or preprocessing version rebuilds
the complete record.

## Model architecture

The PVC trajectory model has the existing clean history and noisy future
trajectory-token stream, and a second view-token stream:

```text
trajectory tokens [B, 49, 2048, D] -- spatial self-attention -- temporal attention -- output head
view tokens       [B, 49, 2048, D] -- spatial self-attention ---------------------- discarded
```

View tokens use the same shared XYZ `PointEmbed` and shared Utonia
normalization/projection as trajectory points. They additionally receive one
learned shared point-view token-type vector and the matching frame's temporal
portion of the current position encoding. They never receive the current
fixed 0--2047 spatial-slot encoding because their indices are not tracked
across frames.

At every transformer block and every frame, concatenate trajectory and view
tokens. Apply the existing spatial AdaLN, full bidirectional self-attention,
and MLP to the joined sequence. The attention key/value mask always enables
all trajectory tokens and enables only valid view tokens; padded view entries
cannot be attended to. Split the streams after the spatial sublayers. View
states remain in the view stream for the next block, but only trajectory
states enter temporal attention and only trajectory states are decoded. Thus
the point views are conditioning only; they are never noised, sampled, or
rendered as predictions.

## Training and sampling

`train_pvc.py` prepares both the existing frame-zero trajectory Utonia cache
and the new full-clip point-view Utonia cache before constructing the loader.
It passes point views, their mask, and their cached Utonia features through
every forward pass.

PVC sampling requires all 49 aligned point views and their masks for every
denoising step. The four clean history frames remain part of the returned
visualization; the 45 future trajectory frames are generated as before.

## Validation

CPU tests cover:

- PVC config rejection of any non-fixed training contract;
- all 49 point-view-file requirements, packing, padding, masks, empty views,
  and malformed/oversized source failures;
- per-sample cache creation, cache reuse, and invalidation after a view source
  changes;
- shared XYZ/Utonia view encoding, temporal-only view positioning, learned
  type embedding, and view input shape validation;
- spatial attention mask behavior, bidirectional joined spatial states, view
  state persistence, and absence of view tokens from temporal attention and
  decoding;
- trainer and sampler forwarding the full view condition on each denoising
  step;
- legacy PC test coverage remaining unchanged.

Run focused PVC tests, then the complete Wan test suite, compilation checks,
and `git diff --check`. A GPU run that prepares the new cache is the manual
integration check.
