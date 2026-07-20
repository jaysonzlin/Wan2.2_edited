# Wan PC Flow Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Wan-native point-cloud trajectory model that trains on PhysCtrl-format HDF5 clips with shifted continuous flow matching and returns 48 absolute future point clouds.

**Architecture:** New PC-specific data, flow, model, pipeline, visualization, and trainer modules live in Wan2.2_edited. The model retains `config_pc.yaml`'s Fourier point embed, factorized joint-spatial/temporal topology, ordered point/time positions, and 8×256 capacity; it replaces DDPM x0 prediction with Wan flow matching and adapted Wan norms, gates, MLPs, head, and UniPC flow solver.

**Tech Stack:** Python 3.10, PyTorch, Accelerate, Diffusers, h5py, NumPy, Matplotlib, ImageIO, PyYAML, pytest.

## Global Constraints

- Modify only Wan2.2_edited; do not import or edit edited-physctrl.
- Do not modify TI2V files: `wan/modules/model.py`, VAE/T5 modules, or `train_i2v.py`.
- Require `(49, 1, 2048, 3)` point clouds and both `(1, 3)` velocity values in every HDF5 sample.
- Production model is 8 layers, width 256, 4 heads, 48 future frames, source frame conditioning, `PointEmbed`, and condition-drop rate 0.
- Train future displacement `d = p_future - p0` with `epsilon - d`; never apply `p0` in the model head.
- Use shifted flow time `t = 5u / (1 + 4u)` and per-frame times `[0, t, ..., t]`.
- Do not copy unused PhysCtrl block variants, physics conditions, PEFT/Diffusers wrappers, DDPM/DDIM code, text conditioning, VAE/video patching, 3D RoPE, or generic visualizer/UI code.
- Add `h5py>=3.10` and `matplotlib>=3.8` to both `requirements.txt` and `pyproject.toml`.

---

### Task 1: Add the owned PC config and strict dataset

**Files:**
- Modify: `requirements.txt`, `pyproject.toml`
- Create: `configs/train/config_pc.yaml`, `training/pc_config.py`, `training/pc_dataset.py`
- Create: `tests/test_pc_config.py`, `tests/test_pc_dataset.py`

**Interfaces:**
- `load_pc_config(path: str | Path, overrides: list[str]) -> dict`
- `PCTrajectoryDataset(dataset_root: str | Path, expected_frames: int = 49, expected_points: int = 2048)`
- Dataset items: `points_src [1,N,3]`, `points_tgt [48,1,N,3]`, both velocity tensors `[1,3]`, and `sample_id`.

- [ ] **Step 1: Write failing config and HDF5 tests**

```python
def test_pc_config_accepts_the_fixed_contract(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  dataset_root: training_dataset\n  num_frames: 49\n  num_points: 2048\nmodel:\n  n_layers: 8\n  latent_dim: 256\n  num_heads: 4\nflow:\n  prediction_type: flow\n  time_shift: 5.0\n")
    assert load_pc_config(path, [])["flow"]["prediction_type"] == "flow"

def test_pc_config_rejects_wrong_frame_count(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  num_frames: 48\n  num_points: 2048\nflow:\n  prediction_type: flow\n  time_shift: 5\n")
    with pytest.raises(ValueError, match="data.num_frames must be 49"):
        load_pc_config(path, [])
```

```python
def test_dataset_splits_a_valid_hdf5_clip(tmp_path):
    write_pc_sample(tmp_path / "sample_0")
    sample = PCTrajectoryDataset(tmp_path)[0]
    assert sample["points_src"].shape == (1, 2048, 3)
    assert sample["points_tgt"].shape == (48, 1, 2048, 3)

def test_dataset_rejects_wrong_point_shape(tmp_path):
    write_pc_sample(tmp_path / "sample_0", shape=(49, 1, 8, 3))
    with pytest.raises(ValueError, match=r"point_cloud must have shape \(49, 1, 2048, 3\)"):
        PCTrajectoryDataset(tmp_path)
```

`write_pc_sample` must create parent directories and HDF5 datasets `point_cloud`, `initial_linear_velocity`, and `initial_angular_velocity` as float32.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_pc_config.py tests/test_pc_dataset.py -q`

Expected: FAIL with missing `training.pc_config` and `training.pc_dataset` imports.

- [ ] **Step 3: Add dependencies and the local config**

Append to `requirements.txt` and add the same strings to `[project].dependencies`:

```text
# Added for point-cloud flow training and trajectory visualization.
h5py>=3.10
matplotlib>=3.8
```

Create `configs/train/config_pc.yaml` with these exact values:

```yaml
output_dir: ./outputs/pc_flow_8layers
logging_dir: logs
vis_dir: vis
report_to: wandb
tracker_project_name: pc_flow
seed: 0
train_batch_size: 1
num_train_epochs: 100
max_train_steps: 60000
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
lr_warmup_steps: 100
adam_beta1: 0.9
adam_beta2: 0.999
adam_weight_decay: 1.0e-2
adam_epsilon: 1.0e-8
max_grad_norm: 1.0
mixed_precision: bf16
dataloader_num_workers: 8
checkpointing_steps: 250
resume_from_checkpoint: null
condition_drop_rate: 0.0
data: {dataset_root: training_dataset, num_frames: 49, num_points: 2048}
model: {n_layers: 8, latent_dim: 256, num_heads: 4, point_embed: true, frame_cond: true, transformer_block: SpatialTemporalTransformerBlock}
flow: {prediction_type: flow, time_shift: 5.0, num_train_timesteps: 1000}
sampling: {num_inference_steps: 50, solver_order: 3}
visualization: {every_epochs: 100, fps: 12}
```

- [ ] **Step 4: Implement the narrow config and dataset modules**

Copy only YAML parsing and `section.key=value` override behavior from `training/overfit_config.py`. `validate_pc_config` must reject anything except 49 frames, 2,048 points, `(n_layers, latent_dim, num_heads) == (8, 256, 4)`, `prediction_type == "flow"`, and positive `time_shift`.

`PCTrajectoryDataset` must discover sorted `sample_*` directories, require at least one, validate every `pc.hdf5` at construction, and read tensors while the HDF5 handle remains open. Missing keys raise `KeyError`; shape errors name the offending dataset and expected exact shape.

- [ ] **Step 5: Verify and commit the data/config slice**

Run: `pytest tests/test_pc_config.py tests/test_pc_dataset.py -q`

Expected: PASS.

```bash
git add requirements.txt pyproject.toml configs/train/config_pc.yaml training/pc_config.py training/pc_dataset.py tests/test_pc_config.py tests/test_pc_dataset.py
git commit -m "feat: add strict PC flow data contract"
```

### Task 2: Add shifted flow batches and a UniPC point-cloud pipeline

**Files:**
- Create: `training/pc_flow.py`, `wan/pc_pipeline.py`
- Create: `tests/test_pc_flow.py`, `tests/test_pc_pipeline.py`

**Interfaces:**
- `PCFlowBatch(model_input, velocity_target, frame_times)`
- `make_pc_flow_batch(future_points, init_pc, generator, time_shift, num_train_timesteps) -> PCFlowBatch`
- `flow_mse(prediction, target) -> Tensor`
- `PCFlowPipeline(model, scheduler, time_shift).__call__(init_pc, initial_linear_velocity, initial_angular_velocity, device, num_inference_steps, generator) -> Tensor[B,48,1,N,3]`

- [ ] **Step 1: Write failing math and sampling tests**

```python
def test_flow_batch_uses_displacements_and_source_time_zero():
    source = torch.full((1, 1, 2, 3), 10.0)
    future = torch.full((1, 48, 1, 2, 3), 11.0)
    batch = make_pc_flow_batch(future, source, torch.Generator().manual_seed(0), 5.0, 1000)
    assert batch.model_input.shape == future.shape
    assert batch.velocity_target.shape == future.shape
    assert torch.equal(batch.frame_times[:, :1], torch.zeros(1, 1))
    assert torch.all(batch.frame_times[:, 1:] > 0)

def test_flow_target_is_noise_minus_displacement(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", lambda tensor, generator=None: torch.full_like(tensor, 3.0))
    batch = make_pc_flow_batch(torch.ones(1, 48, 1, 1, 3), torch.zeros(1, 1, 1, 3), torch.Generator().manual_seed(0), 1.0, 1000)
    assert torch.equal(batch.velocity_target, torch.full((1, 48, 1, 1, 3), 2.0))
```

```python
class ZeroFlowModel(torch.nn.Module):
    n_future_frames = 48
    def forward(self, noisy, frame_times, init_pc, linear, angular):
        return torch.zeros_like(noisy)

def test_pipeline_adds_source_only_after_integration():
    pipeline = PCFlowPipeline(ZeroFlowModel(), FakeFlowScheduler(), time_shift=5.0)
    output = pipeline(torch.full((1, 1, 2, 3), 7.0), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), "cpu", 2, torch.Generator().manual_seed(0))
    assert output.shape == (1, 48, 1, 2, 3)
    assert torch.allclose(output, torch.full_like(output, 7.0))
```

`FakeFlowScheduler.step` returns `SimpleNamespace(prev_sample=sample)` and records the shift supplied to `set_timesteps`.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_pc_flow.py tests/test_pc_pipeline.py -q`

Expected: FAIL because the PC flow modules do not exist.

- [ ] **Step 3: Implement the objective and pipeline**

```python
displacements = future_points - init_pc.unsqueeze(1)
u = torch.rand((future_points.shape[0],), device=future_points.device, dtype=future_points.dtype, generator=generator)
t = time_shift * u / (1 + (time_shift - 1) * u)
noise = torch.randn(displacements.shape, device=displacements.device, dtype=displacements.dtype, generator=generator)
model_input = (1 - t[:, None, None, None, None]) * displacements + t[:, None, None, None, None] * noise
velocity_target = noise - displacements
frame_times = torch.cat((torch.zeros_like(t[:, None]), t[:, None].expand(-1, 48)), dim=1) * num_train_timesteps
```

Validate all dimensions and positive scalar arguments. `flow_mse` must raise `ValueError` for differing prediction/target shapes before calling `F.mse_loss`.

Pipeline sampling must initialize Gaussian displacement frames; call `scheduler.set_timesteps(num_inference_steps, device=device, shift=self.time_shift)`; construct `[0, timestep, ..., timestep]` for every solver call; and return `sample + init_pc.unsqueeze(1)`. Production construction in the trainer must be:

```python
FlowUniPCMultistepScheduler(num_train_timesteps=1000, solver_order=3,
                            prediction_type="flow_prediction", shift=1,
                            use_dynamic_shifting=False)
```

Passing `shift=1` to the constructor and `time_shift` only to `set_timesteps` prevents double shifting.

- [ ] **Step 4: Verify and commit the flow slice**

Run: `pytest tests/test_pc_flow.py tests/test_pc_pipeline.py -q`

Expected: PASS.

```bash
git add training/pc_flow.py wan/pc_pipeline.py tests/test_pc_flow.py tests/test_pc_pipeline.py
git commit -m "feat: add Wan PC flow pipeline"
```

### Task 3: Build the focused factorized PC model

**Files:**
- Create: `wan/modules/pc_flow.py`
- Modify: `wan/modules/__init__.py`
- Create: `tests/test_pc_flow_model.py`

**Interfaces:**
- `PCFlowModel(n_points=2048, n_future_frames=48, latent_dim=256, n_layers=8, num_heads=4, point_embed=True)`
- `forward(noisy_displacements, frame_times, init_pc, initial_linear_velocity, initial_angular_velocity) -> Tensor[B,48,1,N,3]`

- [ ] **Step 1: Write failing model tests**

```python
def make_tiny_model():
    return PCFlowModel(n_points=8, n_future_frames=48, latent_dim=64, n_layers=1, num_heads=1, point_embed=False)

def test_model_returns_direct_future_flow_shape():
    model = make_tiny_model()
    output = model(torch.randn(2, 48, 1, 8, 3), torch.tensor([[0.0] + [500.0] * 48] * 2), torch.randn(2, 1, 8, 3), torch.randn(2, 1, 3), torch.randn(2, 1, 3))
    assert output.shape == (2, 48, 1, 8, 3)

def test_model_rejects_nonzero_source_time():
    model = make_tiny_model()
    with pytest.raises(ValueError, match="frame_times\\[:, 0\\] must be zero"):
        model(torch.zeros(1, 48, 1, 8, 3), torch.ones(1, 49), torch.zeros(1, 1, 8, 3), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))

def test_zero_flow_head_never_adds_source_coordinates():
    model = make_tiny_model()
    torch.nn.init.zeros_(model.flow_head.projection.weight)
    torch.nn.init.zeros_(model.flow_head.projection.bias)
    output = model(torch.zeros(1, 48, 1, 8, 3), torch.tensor([[0.0] + [1.0] * 48]), torch.full((1, 1, 8, 3), 9.0), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))
    assert torch.equal(output, torch.zeros_like(output))
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_pc_flow_model.py -q`

Expected: FAIL because `wan.modules.pc_flow` does not exist.

- [ ] **Step 3: Implement only these selected classes**

Create `PointEmbed`, `PCAdaptiveModulation`, `PCSelfAttention`, `PCSpatialTemporalBlock`, `PCFlowHead`, and `PCFlowModel`. Copy only PhysCtrl's 96-feature Fourier PointEmbed formula (16 frequencies per XYZ axis, sin/cos and raw XYZ then linear projection). Use no other PhysCtrl class.

`PCSelfAttention` has separate q/k/v/o linears, uses `WanRMSNorm` for q and k, reshapes `[B,L,C]` to `[B,L,H,C/H]`, uses `torch.nn.functional.scaled_dot_product_attention`, then applies `o`. Require `latent_dim % num_heads == 0`.

Map each scalar per-frame flow time through a sinusoidal `latent_dim` embedding and `PCAdaptiveModulation` to attention shift/scale/gate and MLP shift/scale/gate. The spatial operation must be:

```python
joint = torch.cat((condition_tokens, point_tokens), dim=1)
joint = joint + gate_attn * attention(layer_norm(joint) * (1 + scale_attn) + shift_attn)
joint = joint + gate_mlp * mlp(layer_norm(joint) * (1 + scale_mlp) + shift_mlp)
condition_tokens, point_tokens = joint[:, :2], joint[:, 2:]
```

Expand the two linear velocity embeddings to `[B,49,2,C]`; spatial blocks update them independently per frame. After every spatial operation, reshape point tokens to `[B*N,49,C]` and apply time-modulated `PCSelfAttention`. Do not add a temporal MLP.

Build fixed non-learned sinusoidal point-index and frame-index tensors and add them to point tokens. Encode the 49-token stream formed from absolute `init_pc` followed by noisy displacement frames. The flow head normalizes/modulates only the 48 future token groups, projects XYZ, and never adds `init_pc`.

Validate exact tensor ranks/shapes and `frame_times.shape == (B,49)`; reject source frame times that are not zero. Export `PCFlowModel` from `wan/modules/__init__.py`.

- [ ] **Step 4: Verify and commit the model**

Run: `pytest tests/test_pc_flow_model.py -q`

Expected: PASS.

```bash
git add wan/modules/pc_flow.py wan/modules/__init__.py tests/test_pc_flow_model.py
git commit -m "feat: add factorized Wan PC flow model"
```

### Task 4: Add the MP4 comparison renderer and Accelerate trainer

**Files:**
- Create: `training/pc_visualization.py`, `train_pc.py`
- Create: `tests/test_pc_visualization.py`, `tests/test_train_pc.py`

**Interfaces:**
- `save_pointcloud_comparison_mp4(prediction, ground_truth, output_path, fps=12) -> None`
- `train_pc.py --config configs/train/config_pc.yaml [section.key=value ...]`

- [ ] **Step 1: Write failing visualization and trainer tests**

```python
def test_comparison_visualization_writes_mp4(tmp_path):
    trajectory = np.zeros((49, 1, 2, 3), dtype=np.float32)
    output = tmp_path / "comparison.mp4"
    save_pointcloud_comparison_mp4(trajectory, trajectory, output, fps=1)
    assert output.is_file() and output.stat().st_size > 0

def test_train_pc_help_is_local_only():
    result = subprocess.run([sys.executable, "train_pc.py", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout

def test_visualization_path_is_inside_configured_vis_directory():
    assert visualization_path(Path("outputs/run"), "vis", 12) == Path("outputs/run/vis/epoch_0012.mp4")

def test_train_pc_runs_one_cpu_step(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pc, "PCTrajectoryDataset", TinyPCDataset)
    train_pc.main(make_tiny_pc_config(tmp_path))
    assert (tmp_path / "output" / "config.yaml").is_file()
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_pc_visualization.py tests/test_train_pc.py -q`

Expected: FAIL because PC visualization and trainer modules do not exist.

- [ ] **Step 3: Implement the minimal renderer**

Validate equal `(frames, objects, points, 3)` arrays. Render predicted and target 3D scatter subplots with one shared cubic extent, stable `viridis` colors derived from each point's source-frame Z height, and ImageIO MP4 output. Always close the writer and matplotlib figure in `finally`. Do not port PhysCtrl's interactive CLI, standalone single-cloud renderer, or UI code.

- [ ] **Step 4: Implement the trainer core**

Keep `train_pc.py` top-level imports limited to argparse, pathlib, and `load_pc_config` so `--help` does not load CUDA/W&B dependencies. In `main()`, lazily import Accelerate, model/pipeline, flow helpers, dataset, visualizer, DataLoader, and cosine scheduler.

The accumulated update must be:

```python
flow_batch = make_pc_flow_batch(batch["points_tgt"].to(accelerator.device), batch["points_src"].to(accelerator.device), generator, config["flow"]["time_shift"], config["flow"]["num_train_timesteps"])
prediction = model(flow_batch.model_input, flow_batch.frame_times, batch["points_src"].to(accelerator.device), batch["initial_linear_velocity"].to(accelerator.device), batch["initial_angular_velocity"].to(accelerator.device))
loss = flow_mse(prediction, flow_batch.velocity_target)
```

`TinyPCDataset` must return a single zero-valued item with `points_src [1,8,3]`, `points_tgt [48,1,8,3]`, and both velocity tensors `[1,3]`; `make_tiny_pc_config` must override output directory, batch size 1, one layer/64-wide model, one optimizer step, CPU/no mixed precision, no reporting, no workers, and disable visualization. This keeps the smoke test local while the production YAML remains fixed.

Write the resolved YAML config to `output_dir/config.yaml`; use the YAML AdamW/cosine warmup settings; log loss and learning rate; clip gradients; save state every `checkpointing_steps`; and reuse the existing Wan `latest` checkpoint-fallback behavior. On the configured visualization epoch, use a deterministic one-item `PCFlowPipeline` sample and compare `torch.cat((points_src.unsqueeze(1), future_prediction), dim=1)` with its complete 49-frame ground truth.

- [ ] **Step 5: Verify and commit the runnable workflow**

Run: `pytest tests/test_pc_visualization.py tests/test_train_pc.py -q`

Expected: PASS.

```bash
git add training/pc_visualization.py train_pc.py tests/test_pc_visualization.py tests/test_train_pc.py
git commit -m "feat: add Wan PC flow trainer"
```

### Task 5: Document the workflow and verify all regressions

**Files:**
- Modify: `README.md`, `tests/README.md`, `tests/test_train_pc.py`

- [ ] **Step 1: Add a failing documentation acceptance test**

```python
def test_readme_documents_pc_flow_entrypoint():
    readme = Path("README.md").read_text()
    assert "train_pc.py --config configs/train/config_pc.yaml" in readme
    assert "pc.hdf5" in readme
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_train_pc.py::test_readme_documents_pc_flow_entrypoint -q`

Expected: FAIL until the documented command exists.

- [ ] **Step 3: Add concise run instructions**

Add a `Point-cloud flow training` section near the current Kubric overfit section in `README.md`:

```bash
accelerate launch --config_file configs/accelerate/h200_single_gpu.yaml \
  train_pc.py --config configs/train/config_pc.yaml
```

Document the three HDF5 datasets and exact shapes, the `training_dataset/sample_*/pc.hdf5` location, checkpoints below `outputs/pc_flow_8layers`, and comparison MP4s below its `vis/` directory. Add the focused PC pytest command to `tests/README.md`.

- [ ] **Step 4: Run PC and TI2V regression verification**

Run: `pytest tests/test_pc_config.py tests/test_pc_dataset.py tests/test_pc_flow.py tests/test_pc_pipeline.py tests/test_pc_flow_model.py tests/test_pc_visualization.py tests/test_train_pc.py -q`

Expected: PASS.

Run: `pytest tests/test_overfit_config.py tests/test_overfit_dataset.py tests/test_wan_i2v_training.py tests/test_train_i2v.py -q`

Expected: PASS; TI2V stays untouched.

Run: `python -c "from training.pc_dataset import PCTrajectoryDataset; print(len(PCTrajectoryDataset('training_dataset')))"`

Expected: a positive integer once `h5py` is installed.

- [ ] **Step 5: Commit the docs and verification test**

```bash
git add README.md tests/README.md tests/test_train_pc.py
git commit -m "docs: document Wan PC flow training"
```

## Self-review

- Spec coverage: tasks cover owned config/data, strict HDF5, selected model topology, shifted flow objective, UniPC sampling, visualization, training, dependencies, docs, and TI2V isolation.
- No-surplus port: the model task creates only the six selected PC classes and explicitly excludes PhysCtrl's generic wrappers and optional systems.
- Interface consistency: dataset outputs feed batch creation and trainer; the model signature is shared by trainer and pipeline; pipeline returns absolute 48 future frames used by the 49-frame comparison renderer.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-wan-pc-flow-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task and review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
