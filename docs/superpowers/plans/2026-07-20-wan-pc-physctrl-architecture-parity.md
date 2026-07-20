# Wan PC PhysCtrl Architecture Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Wan's point-cloud trajectory backbone as a dependency-free, from-scratch PhysCtrl PC-DiT-equivalent architecture while retaining Wan's DDPM and optional flow workflows.

**Architecture:** Put the small, reusable PhysCtrl-equivalent modules in `wan/modules/pc_physctrl.py` and leave `wan/modules/pc_trajectory.py` as the public objective adapter and trajectory wrapper.  Default DDPM embeds an identical batch timestep at every source/future frame and predicts x0 offsets; flow retains its existing absolute-input and direct-vector adapter.  Tests use literal PyTorch reference equations rather than importing PhysCtrl or Diffusers.

**Tech Stack:** Python 3, PyTorch (`torch.nn`, `torch.nn.functional.scaled_dot_product_attention`), pytest, existing Accelerate/Diffusers schedulers only outside the backbone.

## Global Constraints

- Do not import Diffusers, CogVideoX, PEFT, or `WanRMSNorm` from either `wan/modules/pc_physctrl.py` or `wan/modules/pc_trajectory.py`.
- Keep the public `PCTrajectoryModel.forward(noisy_future_state, frame_times, init_pc, initial_linear_velocity, initial_angular_velocity)` signature and existing pipelines/trainer/data APIs.
- Keep `configs/train/config_pc.yaml` structurally unchanged; enforce its active PhysCtrl-equivalent model settings through validation.
- The default DDPM path must use absolute noisy positions and return absolute x0 after the PhysCtrl-style source offset; the flow path must retain its source-time-zero, displacement-output behavior.
- Implement only the active `SpatialTemporalTransformerBlock` path: no dropout modules, CFG controls, LoRA/PEFT, processor swapping/fusion, RoPE, text/class/ofs conditions, alternative blocks, patching, or temporal compression.
- Use TDD: run each newly added test and observe its expected failure before production implementation.

---

## File structure

- Create `wan/modules/pc_physctrl.py`: dependency-free positional embedding, learned timestep embedding, attention, adaptive norms, feed-forward, block, and output-head primitives.
- Modify `wan/modules/pc_trajectory.py`: make `PCTrajectoryModel` compose the new primitives while preserving its input/output contract and objective adapters.
- Modify `training/pc_config.py`: validate that the existing YAML represents exactly the active PhysCtrl PC configuration.
- Modify `tests/test_pc_config.py`: extend the valid fixture and test all new configuration rejections.
- Create `tests/test_pc_physctrl_components.py`: deterministic primitive and block-reference parity tests.
- Modify `tests/test_pc_trajectory_model.py`: retain flow/DDPM contract coverage against the rebuilt architecture and assert the inactive module graph is absent.

### Task 1: Lock the configuration to the active PhysCtrl PC architecture

**Files:**
- Modify: `training/pc_config.py:31-53`
- Modify: `tests/test_pc_config.py:1-58`

**Interfaces:**
- Consumes: the existing nested `dict` produced by `load_pc_config(path, overrides)`.
- Produces: `validate_pc_config(config: dict) -> None`, which accepts the current `configs/train/config_pc.yaml` values and raises `ValueError` for any non-parity model setting.

- [ ] **Step 1: Write failing configuration tests**

Replace the repeated inline YAML with a helper that contains the current model fields, then add these explicit rejection cases:

```python
def valid_config_text(model_lines="") -> str:
    return (
        "data:\n  dataset_root: training_dataset\n  num_frames: 49\n  num_points: 2048\n"
        "model:\n  n_layers: 8\n  latent_dim: 256\n  num_heads: 4\n"
        "  point_embed: true\n  frame_cond: true\n"
        "  transformer_block: SpatialTemporalTransformerBlock\n"
        f"{model_lines}"
        "objective:\n  type: ddpm\n  num_train_timesteps: 1000\n"
        "  beta_schedule: linear\n  time_shift: 5.0\n"
        "lr_scheduler: cosine\n"
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("  point_embed: false\n", "model.point_embed must be true"),
        ("  frame_cond: false\n", "model.frame_cond must be true"),
        ("  transformer_block: TemporalOnlyTransformerBlock\n", "model.transformer_block"),
    ],
)
def test_pc_config_rejects_non_physctrl_model_option(tmp_path, replacement, message):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text().replace(
        "  transformer_block: SpatialTemporalTransformerBlock\n",
        replacement if "transformer_block" in replacement
        else "  transformer_block: SpatialTemporalTransformerBlock\n" + replacement,
    ))
    with pytest.raises(ValueError, match=message):
        load_pc_config(path, [])
```

Add a separate `num_heads: 2` fixture assertion with the message
`"model.num_heads must equal model.latent_dim // 64"` so this relationship is
tested independently from the existing fixed-shape tuple check.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_pc_config.py -q`

Expected: FAIL because `validate_pc_config` does not yet require
`point_embed`, `frame_cond`, the active block type, or the head-width
relationship.

- [ ] **Step 3: Implement the exact validation rules**

After the current fixed `(8, 256, 4)` model check in `validate_pc_config`, add:

```python
if model.get("num_heads") != model.get("latent_dim", 0) // 64:
    raise ValueError("model.num_heads must equal model.latent_dim // 64")
if model.get("point_embed") is not True:
    raise ValueError("model.point_embed must be true")
if model.get("frame_cond") is not True:
    raise ValueError("model.frame_cond must be true")
if model.get("transformer_block") != "SpatialTemporalTransformerBlock":
    raise ValueError(
        "model.transformer_block must be 'SpatialTemporalTransformerBlock'"
    )
```

Do not alter `configs/train/config_pc.yaml`; it already supplies all four
accepted values.

- [ ] **Step 4: Run the configuration tests to verify they pass**

Run: `python -m pytest tests/test_pc_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add training/pc_config.py tests/test_pc_config.py
git commit -m "test: lock Wan PC config to PhysCtrl architecture"
```

### Task 2: Add exact positional and diffusion-time primitives

**Files:**
- Create: `wan/modules/pc_physctrl.py`
- Create: `tests/test_pc_physctrl_components.py`

**Interfaces:**
- Produces `physctrl_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor`.
- Produces `physctrl_position_embedding(num_points: int, num_frames: int, dim: int) -> torch.Tensor` with shape `(1, 2 + num_frames * num_points, dim)`.
- Produces `PhysCtrlTimestepEmbedding(dim: int)`, whose `forward(timesteps: torch.Tensor) -> torch.Tensor` accepts any leading shape and appends the feature dimension.

- [ ] **Step 1: Write failing primitive parity tests**

Create tests that compute the expected values directly rather than importing
the existing Wan sinusoid helper:

```python
def reference_1d(positions, dim):
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim / 2)
    angles = positions.reshape(-1, 1).to(torch.float64) / (10000**omega)
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.float32)


def test_position_embedding_uses_physctrl_64_192_channel_split():
    position = physctrl_position_embedding(num_points=3, num_frames=2, dim=256)
    assert position.shape == (1, 8, 256)
    assert torch.equal(position[:, :2], torch.zeros_like(position[:, :2]))
    expected = torch.cat((
        reference_1d(torch.arange(2).repeat_interleave(3), 64),
        reference_1d(torch.arange(3).repeat(2), 192),
    ), dim=-1)
    torch.testing.assert_close(position[0, 2:], expected)


def test_timestep_embedding_uses_cogvideox_cos_then_sin_frequencies():
    module = PhysCtrlTimestepEmbedding(8)
    with torch.no_grad():
        module.linear_1.weight.copy_(torch.eye(8)); module.linear_1.bias.zero_()
        module.linear_2.weight.copy_(torch.eye(8)); module.linear_2.bias.zero_()
    times = torch.tensor([[0.0, 2.0]])
    half = 4
    frequency = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / (half - 1))
    raw = torch.cat(((times[..., None] * frequency).cos(), (times[..., None] * frequency).sin()), dim=-1)
    torch.testing.assert_close(module(times), torch.nn.functional.silu(raw), atol=1e-6, rtol=1e-6)
```

- [ ] **Step 2: Run the primitive tests to verify they fail**

Run: `python -m pytest tests/test_pc_physctrl_components.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'wan.modules.pc_physctrl'`.

- [ ] **Step 3: Implement the primitive module**

Create `wan/modules/pc_physctrl.py` with the following exact public code
shape.  Use float32 buffer construction, preserving the input dtype on return.

```python
def physctrl_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("dim must be even")
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float64)
    omega = 1.0 / 10000 ** (omega / (dim / 2.0))
    angles = positions.reshape(-1, 1).to(torch.float64) * omega
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.float32)


def physctrl_position_embedding(num_points: int, num_frames: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("dim must be divisible by 4")
    temporal = physctrl_1d_sincos(torch.arange(num_frames).repeat_interleave(num_points), dim // 4)
    spatial = physctrl_1d_sincos(torch.arange(num_points).repeat(num_frames), 3 * dim // 4)
    points = torch.cat((temporal, spatial), dim=-1)
    return torch.cat((torch.zeros(1, 2, dim), points.unsqueeze(0)), dim=1)


class PhysCtrlTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1))
        angles = timesteps.float()[..., None] * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1).to(timesteps.dtype)
        return self.linear_2(self.act(self.linear_1(embedding)))
```

Import `math`, `torch`, and `torch.nn as nn`; do not import Wan model code or
Diffusers.  Keep `physctrl_position_embedding` on CPU when called by model
construction, then register it as a non-persistent buffer in the wrapper task.

- [ ] **Step 4: Run the primitive tests to verify they pass**

Run: `python -m pytest tests/test_pc_physctrl_components.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the primitive implementation**

```bash
git add wan/modules/pc_physctrl.py tests/test_pc_physctrl_components.py
git commit -m "feat: add PhysCtrl PC positional and timestep primitives"
```

### Task 3: Rebuild the PhysCtrl block from basic PyTorch components

**Files:**
- Modify: `wan/modules/pc_physctrl.py`
- Modify: `tests/test_pc_physctrl_components.py`

**Interfaces:**
- Produces `PhysCtrlAttention(dim: int, heads: int)`, `PhysCtrlLayerNormZero(dim: int)`, `PhysCtrlAdaLayerNorm(dim: int)`, and `PhysCtrlSpatialTemporalBlock(dim: int, heads: int)`.
- `PhysCtrlSpatialTemporalBlock.forward(points, controls, temb)` consumes `(B, F, N, C)`, `(B, F, 2, C)`, and `(B, F, C)` and returns points and controls with those same shapes.

- [ ] **Step 1: Write failing block-equation tests**

Add three deterministic tests.  Set all unspecified parameters to zero and
use `C=4`, `H=2` so every expected tensor can be calculated explicitly:

```python
def test_attention_normalizes_q_and_k_per_head_before_sdpa():
    attention = PhysCtrlAttention(dim=4, heads=2)
    set_identity_qkvo(attention)
    tokens = torch.tensor([[[1.0, 3.0, 2.0, 6.0], [2.0, 4.0, 4.0, 8.0]]])
    q = torch.nn.functional.layer_norm(tokens.reshape(1, 2, 2, 2), (2,), attention.q_norm.weight, attention.q_norm.bias, 1e-6).transpose(1, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, q, tokens.reshape(1, 2, 2, 2).transpose(1, 2)).transpose(1, 2).reshape(1, 2, 4)
    torch.testing.assert_close(attention(tokens), expected)


def test_layer_norm_zero_uses_distinct_point_and_control_modulation():
    module = PhysCtrlLayerNormZero(4)
    configure_distinct_six_chunks(module.linear)
    points, controls = torch.ones(1, 1, 4), torch.ones(1, 2, 4)
    point_out, control_out, point_gate, control_gate = module(points, controls, torch.ones(1, 4))
    assert not torch.equal(point_out, control_out[:, :1])
    assert not torch.equal(point_gate, control_gate)


def test_temporal_path_is_adaln_then_ungated_attention_residual():
    block = PhysCtrlSpatialTemporalBlock(dim=4, heads=2)
    zero_spatial_and_mlp(block); set_identity_temporal_attention(block.temporal_attention)
    points = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0, 5.0]]]])
    controls = torch.zeros(1, 2, 2, 4)
    temb = torch.zeros(1, 2, 4)
    output, _ = block(points, controls, temb)
    expected = reference_temporal_adaln_residual(points, block.temporal_norm, block.temporal_attention, temb)
    torch.testing.assert_close(output, expected)
```

Define `set_identity_qkvo`, `configure_distinct_six_chunks`,
`zero_spatial_and_mlp`, and `reference_temporal_adaln_residual` in the test
file.  They must operate directly on real parameters and PyTorch math, not
mocks or an imported PhysCtrl implementation.

- [ ] **Step 2: Run the block tests to verify they fail**

Run: `python -m pytest tests/test_pc_physctrl_components.py -q`

Expected: FAIL with import errors for the four missing block classes.

- [ ] **Step 3: Implement the exact basic modules**

Add the following implementations to `pc_physctrl.py`:

```python
class PhysCtrlAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim); self.to_out = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        def heads_of(projection):
            return projection(tokens).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads_of(self.to_q), heads_of(self.to_k), heads_of(self.to_v)
        output = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


class PhysCtrlLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU(); self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, points, controls, temb):
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.act(temb)).chunk(6, dim=-1)
        points = self.norm(points) * (1 + scale[:, None]) + shift[:, None]
        controls = self.norm(controls) * (1 + enc_scale[:, None]) + enc_shift[:, None]
        return points, controls, gate[:, None], enc_gate[:, None]
```

Implement `PhysCtrlAdaLayerNorm` as affine `LayerNorm(C, eps=1e-5)` with a
`SiLU`/`Linear(C, 2C)` controller.  It must accept both `(B, C)` and `(B, L,
C)` `temb`: reshape the two chunks to `(B, 1, C)` or use them directly,
respectively.  This generalization is needed only for flow's source-time-zero
frame; when DDPM supplies one identical timestep across frames it is exactly
the PhysCtrl equation.

Implement the block in this exact order:

```python
flat_points = points.reshape(B * F, N, C)
flat_controls = controls.reshape(B * F, 2, C)
flat_temb = temb.reshape(B * F, C)

p, c, p_gate, c_gate = self.norm1(flat_points, flat_controls, flat_temb)
joined = torch.cat((c, p), dim=1)
joined = joined + torch.cat((c_gate, p_gate), dim=1) * self.spatial_attention(joined)
c, p = joined[:, :2], joined[:, 2:]
p, c, p_gate, c_gate = self.norm2(p, c, flat_temb)
joined = torch.cat((c, p), dim=1)
joined = joined + torch.cat((c_gate, p_gate), dim=1) * self.mlp(joined)
```

Use `nn.Sequential(nn.Linear(C, 4*C), nn.GELU(approximate="tanh"),
nn.Linear(4*C, C))` for `self.mlp`.  Then reshape points as `(B*N, F, C)`,
repeat `temb` over points, apply `temporal_norm`, temporal attention, and the
ungated residual; reshape to `(B, F, N, C)`.  Return the simultaneously
updated controls as `(B, F, 2, C)`.  Do not instantiate `nn.Dropout`.

- [ ] **Step 4: Run the component tests to verify they pass**

Run: `python -m pytest tests/test_pc_physctrl_components.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the rebuilt block**

```bash
git add wan/modules/pc_physctrl.py tests/test_pc_physctrl_components.py
git commit -m "feat: rebuild PhysCtrl PC transformer block"
```

### Task 4: Replace the public Wan trajectory wrapper with the parity backbone

**Files:**
- Modify: `wan/modules/pc_physctrl.py`
- Modify: `wan/modules/pc_trajectory.py:1-143`
- Modify: `tests/test_pc_trajectory_model.py:1-98`

**Interfaces:**
- Produces `PhysCtrlOutputHead(dim: int)` with `forward(points: Tensor, temb: Tensor) -> Tensor`, consuming `(B, F, N, C)` points and `(B, F, C)` timestep embeddings.
- Preserves `PCTrajectoryModel(...)` constructor arguments and `forward(...)` return shape `(B, 48, 1, N, 3)`.

- [ ] **Step 1: Write failing wrapper and graph tests**

Update `make_tiny_model` to pass `point_embed=True`; the physical Fourier
encoder works for eight points and must be used in tests.  Retain the current
five flow/DDPM tests, then add:

```python
def test_default_ddpm_reuses_one_timestep_embedding_at_all_49_frames():
    model = make_tiny_model(objective_type="ddpm")
    captured = {}
    handle = model.time_embedding.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("temb", output.detach().clone())
    )
    try:
        model(torch.zeros(1, 48, 1, 8, 3), torch.full((1, 49), 123.0),
              torch.zeros(1, 1, 8, 3), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))
    finally:
        handle.remove()
    assert captured["temb"].shape == (1, 49, 64)
    assert torch.equal(captured["temb"][:, :1].expand_as(captured["temb"]), captured["temb"])


def test_parity_backbone_has_no_dropout_or_wan_rms_norm_modules():
    model = make_tiny_model(objective_type="ddpm")
    names = {type(module).__name__ for module in model.modules()}
    assert "Dropout" not in names
    assert "WanRMSNorm" not in names
```

Add an end-to-end deterministic test with one block, `C=64`, `N=2`, and
manually set weights.  Compute the expected result by invoking only the
literal reference helpers from `test_pc_physctrl_components.py` (copied into
this test module, not imported from it): point Fourier projection, position
addition, two control tokens, block equations, final LayerNorm, AdaLN, XYZ
projection, and source residual.  Assert close with `atol=rtol=1e-5`.

- [ ] **Step 2: Run the trajectory tests to verify they fail**

Run: `python -m pytest tests/test_pc_trajectory_model.py -q`

Expected: FAIL because the existing model does not expose `time_embedding`,
uses Wan RMSNorm, and contains the old non-PhysCtrl block.

- [ ] **Step 3: Implement the output head and wrapper**

Add this output module to `pc_physctrl.py`:

```python
class PhysCtrlOutputHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-5)
        self.norm_out = PhysCtrlAdaLayerNorm(dim)
        self.projection = nn.Linear(dim, 3)

    def forward(self, points: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm_out(self.norm_final(points), temb))
```

Rewrite `PCTrajectoryModel` to create `PointEmbed`, the two existing velocity
`Linear(3, C)` encoders, `time_embedding = PhysCtrlTimestepEmbedding(C)`,
eight-or-configured-count `PhysCtrlSpatialTemporalBlock(C, num_heads)`, and
`output_head = PhysCtrlOutputHead(C)`.  Register
`physctrl_position_embedding(n_points, n_future_frames + 1, C)` with
`persistent=False`.

Its forward core must be:

```python
future = init_pc.unsqueeze(1) + noisy_future_state if self.objective_type == "flow" else noisy_future_state
coordinates = torch.cat((init_pc.unsqueeze(1), future), dim=1).squeeze(2)
points = self.input_encoder(coordinates.reshape(-1, self.n_points, 3)).reshape(B, 49, self.n_points, C)
points = points + self.position_embedding[:, 2:].to(points).reshape(1, 49, self.n_points, C)
controls = torch.stack((self.linear_velocity_encoder(linear), self.angular_velocity_encoder(angular)), dim=1)
controls = controls[:, None].expand(-1, 49, -1, -1)
temb = self.time_embedding(frame_times)
for block in self.blocks:
    points, controls = block(points, controls, temb)
offset = self.output_head(points[:, 1:], temb[:, 1:]).unsqueeze(2)
return offset if self.objective_type == "flow" else offset + init_pc.unsqueeze(1)
```

Keep all present shape checks.  Preserve the flow-only validation that source
time is zero.  Add `latent_dim % 64 == 0` and `num_heads == latent_dim // 64`
constructor checks with clear errors; tiny tests use `(64, 1)` and the default
configuration uses `(256, 4)`.

- [ ] **Step 4: Run trajectory and component tests to verify they pass**

Run: `python -m pytest tests/test_pc_physctrl_components.py tests/test_pc_trajectory_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the parity wrapper**

```bash
git add wan/modules/pc_physctrl.py wan/modules/pc_trajectory.py \
  tests/test_pc_physctrl_components.py tests/test_pc_trajectory_model.py
git commit -m "feat: use PhysCtrl architecture for Wan PC trajectories"
```

### Task 5: Verify unchanged training and sampling integrations

**Files:**
- Modify only if a failing integration test demonstrates an interface mismatch: `train_pc.py` or `wan/pc_pipeline.py`
- Test: `tests/test_pc_ddpm.py`
- Test: `tests/test_pc_objectives.py`
- Test: `tests/test_pc_pipeline.py`
- Test: `tests/test_train_pc.py`

**Interfaces:**
- Consumes the unchanged `PCTrajectoryModel.forward` contract from Task 4.
- Produces a passing DDPM default train/sampling path and unchanged flow/UniPC path without any backbone-specific branch in the trainer or pipelines.

- [ ] **Step 1: Add a DDPM trainer integration assertion**

In `tests/test_train_pc.py`, extend the existing DDPM configuration test to
assert that `make_pc_ddpm_batch` supplies a `(B, 49)` `frame_times` tensor
with identical entries and that a one-step model invocation returns the
absolute target shape:

```python
assert batch.frame_times.shape == (1, 49)
assert torch.equal(batch.frame_times[:, :1].expand_as(batch.frame_times), batch.frame_times)
assert prediction.shape == batch.target.shape == (1, 48, 1, 8, 3)
```

In `tests/test_pc_pipeline.py`, add a `PCDDIMPipeline` fake-model test that
records every `frame_times` argument and asserts all 49 entries equal the
scheduler timestep.  Keep the existing flow test asserting source time zero
and final source addition.

- [ ] **Step 2: Run integration tests to verify the expected pre-integration failure**

Run: `python -m pytest tests/test_pc_ddpm.py tests/test_pc_objectives.py tests/test_pc_pipeline.py tests/test_train_pc.py -q`

Expected: FAIL only if the rebuilt public model exposes an integration
mismatch; otherwise the newly added assertions must already pass and no
production edit is required.

- [ ] **Step 3: Make only interface-preserving fixes if a test fails**

Do not alter objective math, scheduler selection, visualization, dataset
shapes, or YAML fields.  The only permitted production corrections are:

```python
# train_pc.py construction must remain argument-compatible
model = PCTrajectoryModel(
    n_points=config["data"]["num_points"], n_future_frames=48,
    latent_dim=model_config["latent_dim"], n_layers=model_config["n_layers"],
    num_heads=model_config["num_heads"], point_embed=model_config["point_embed"],
    objective_type=objective["type"],
)
```

and forwarding the existing `frame_times` unmodified into the model.  If no
test exposes an incompatibility, make no production-file edit in this task.

- [ ] **Step 4: Run the focused PC regression suite**

Run: `python -m pytest tests/test_pc_config.py tests/test_pc_dataset.py tests/test_pc_ddpm.py tests/test_pc_objectives.py tests/test_pc_pipeline.py tests/test_pc_trajectory_model.py tests/test_pc_physctrl_components.py tests/test_pc_visualization.py tests/test_train_pc.py -q`

Expected: PASS.

- [ ] **Step 5: Commit integration tests and any required compatibility fix**

```bash
git add tests/test_pc_ddpm.py tests/test_pc_objectives.py tests/test_pc_pipeline.py \
  tests/test_train_pc.py train_pc.py wan/pc_pipeline.py
git commit -m "test: verify PhysCtrl-parity Wan PC integrations"
```

Omit unchanged production paths from `git add`; the command is a target list,
not authorization to create unrelated modifications.

## Plan self-review

- **Spec coverage:** Task 1 enforces the configuration contract; Tasks 2–4 implement every retained encoder, positional, timestep, normalization, attention, MLP, residual, and output component; Tasks 4–5 preserve both objective adapters and public integrations; all listed removed capabilities are excluded by the architecture and module-graph test.
- **No external parity dependency:** Tasks 2–4 calculate expected tensors directly with PyTorch and do not import PhysCtrl or Diffusers.
- **Type consistency:** `PhysCtrlTimestepEmbedding` returns `(B, F, C)` from `(B, F)`; blocks consume and return `(B, F, N, C)` points plus `(B, F, 2, C)` controls; the output head consumes future `(B, 48, N, C)` tokens and returns `(B, 48, N, 3)` before the wrapper restores the object axis.
- **Scope:** No checkpoint conversion, YAML migration, data/sampling/visualization redesign, or non-PC Wan model change is included.
