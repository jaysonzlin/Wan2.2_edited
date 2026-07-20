# Wan PC PhysCtrl Architecture Parity Design

## Goal

Replace Wan PC's custom `PCTrajectoryModel` transformer with a dependency-free
PyTorch implementation of the active PhysCtrl PC-DiT architecture.  The
default DDPM/x0 path must match PhysCtrl's configured architecture and
behavior, while Wan retains its optional flow-matching/UniPC input-output
adapter.

The model is trained from scratch.  Loading PhysCtrl checkpoints, matching its
state-dict names, and retaining any Diffusers or CogVideoX runtime dependency
are out of scope.

## Fixed configuration contract

Keep Wan's current YAML layout.  `validate_pc_config` must reject a PC model
configuration unless it represents the active PhysCtrl setup:

- `data.num_frames == 49` and `data.num_points == 2048`;
- `model.n_layers == 8`, `model.latent_dim == 256`, and
  `model.num_heads == 4`;
- `model.num_heads == model.latent_dim // 64`;
- `model.point_embed is true` and `model.frame_cond is true`;
- `model.transformer_block == "SpatialTemporalTransformerBlock"`.

The existing config needs no schema migration.  `frame_cond` is a validated
contract rather than a source-less-model branch.  The model always constructs
the 49 token groups as clean source frame plus 48 future frames.

## Dependency-free model

`wan/modules/pc_trajectory.py` remains the public module, but its internals
are reconstructed from basic PyTorch layers only.  It may use `torch`,
`torch.nn`, `torch.nn.functional`, and the existing project dependencies; it
must not import Diffusers, CogVideoX, PEFT, or Wan's RMSNorm implementation.

The retained model components are:

1. The existing 96-feature Fourier XYZ encoder and a `Linear(99, C)` point
   projection.
2. Two `Linear(3, C)` tokens for initial linear and angular velocity.
3. PhysCtrl's non-learned positional allocation: 64 channels of frame-index
   sinusoid concatenated with 192 channels of point-index sinusoid.  The two
   velocity tokens receive zeros, and the point tokens receive the flattened
   49-by-2048 position embedding.
4. A learned diffusion timestep embedding:
   fixed sinusoid of width `C`, then `Linear(C, C)`, SiLU, and `Linear(C, C)`.
5. Eight `SpatialTemporalTransformerBlock` instances described below.
6. A final `LayerNorm(C)`, time-conditioned AdaLayerNorm, and `Linear(C, 3)`
   head.  The head processes only future point tokens.

Remove all inactive CogVideoX generality: dropout modules, LoRA/PEFT hooks,
attention processor replacement and fusion, RoPE/image inputs, text/class/ofs
conditioning, alternate transformer blocks, patching and temporal
compression, and classifier-free PC control handling.

## Rebuilt block components

For width `C=256`, heads `H=4`, and head width `D=64`, every block contains:

- Two independent `CogVideoXLayerNormZero` modules.  Each is `SiLU` followed
  by `Linear(C, 6C)` and one affine `LayerNorm(C, eps=1e-5)`.  It produces
  distinct `(shift, scale, gate)` values for point and velocity-control tokens
  even though the LayerNorm weights are shared.
- A joint spatial attention module.  It concatenates the two controls before
  the point tokens, applies biased Q/K/V/O `Linear(C, C)` projections, reshapes
  to heads, performs affine `LayerNorm(D, eps=1e-6)` on Q and K *per head*,
  then calls `scaled_dot_product_attention`.  It splits the result back into
  controls and points after the output projection.
- A shared spatial feed-forward operation on the concatenated token stream:
  `Linear(C, 4C)`, tanh-approximate GELU, and `Linear(4C, C)`.  It has no
  dropout module.
- A temporal AdaLayerNorm: `SiLU`, `Linear(C, 2C)`, and affine
  `LayerNorm(C, eps=1e-5)`.  It normalizes and shift/scales only point tracks.
- A temporal self-attention module with the same biased Q/K/V/O projections
  and per-head Q/K LayerNorm as spatial attention.  It attends over 49 frames
  for each point; controls do not enter this operation.

The spatial path performs separate gated residual updates for points and
controls after both attention and MLP.  The temporal residual is ungated.

## Objective adapters and public API

Keep the current public signature:

```python
forward(noisy_future_state, frame_times, init_pc,
        initial_linear_velocity, initial_angular_velocity)
```

`frame_times` remains `(B, 49)`.  It is embedded per frame before the
corresponding spatial operation and per point-track before the temporal
operation.

- DDPM remains the default.  The trainer supplies a single sampled integer
  timestep expanded to all 49 entries, noisy **absolute** target positions,
  and an x0 loss.  The output head predicts an offset and adds `init_pc`,
  matching PhysCtrl's configured `pred_offset: true` behavior.
- Flow remains optional.  The existing adapter keeps a zero source time,
  common future flow time, and converts displacement state to absolute
  coordinates before embedding.  It returns the direct velocity/displacement
  prediction without adding `init_pc`.

The trainer, data contract, DDIM DDPM pipeline, flow/UniPC pipeline, and
visualization stay unchanged apart from constructing the rebuilt backbone.

## Verification

Implement tests before production changes.  They must use tiny dimensions
where possible and deterministic manually assigned weights.

1. Unit tests verify the 64/192 positional-channel split, learned timestep
   embedding, per-head Q/K LayerNorm, joint attention ordering, separate
   point/control gates, temporal AdaLayerNorm, no-dropout module graph, and
   final two-normalization output path against independently written reference
   tensor calculations.
2. An end-to-end tiny-model reference test sets deterministic weights and
   verifies the rebuilt block/model output against a direct reference sequence
   of the PhysCtrl equations.
3. Existing DDPM and flow contract tests remain: source insertion, absolute
   DDPM x0 residual, flow absolute-input conversion, zero source flow time,
   and pipeline output coordinates.
4. Configuration tests reject every incompatible model setting named in the
   fixed configuration contract while accepting the existing YAML.

No test imports the PhysCtrl repository or Diffusers.  The references are
small literal PyTorch calculations so the Wan test suite remains
self-contained.

## Non-goals

- Checkpoint conversion or checkpoint compatibility with PhysCtrl.
- Recreating unused PhysCtrl model variants or their generic runtime APIs.
- Changing the data format, optimizer, scheduler choices, sampling step
  counts, visualization output, or non-PC Wan video architecture.
