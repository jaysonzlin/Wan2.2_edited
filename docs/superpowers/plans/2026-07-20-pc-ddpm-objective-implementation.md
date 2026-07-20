# PC DDPM Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable PhysCtrl-style DDPM x0 training and DDIM sampling to the Wan PC workflow, selecting DDPM by default.

**Architecture:** The existing factorized PC backbone gets an objective mode. DDPM batch construction and DDIM sampling live in focused units; the trainer selects them or the retained flow/UniPC units.

**Tech Stack:** Python 3.10, PyTorch, Diffusers, pytest.

## Global Constraints

- `objective.type` accepts `flow` and `ddpm`; `config_pc.yaml` selects `ddpm`.
- Flow retains displacement state, flow target, source time zero, and UniPC.
- DDPM uses noisy absolute positions, x0 MSE, and DDIM.
- Preserve PointEmbed, 8×256×4 factorization, HDF5 data, and two velocity controls.
- Do not import PhysCtrl/CogVideoX code.

---

### Task 1: Add and test selectable DDPM behavior

**Files:**
- Create: `training/pc_ddpm.py`, `tests/test_pc_ddpm.py`
- Modify: `configs/train/config_pc.yaml`, `training/pc_config.py`, `wan/modules/pc_flow.py`, `wan/pc_pipeline.py`, `train_pc.py`
- Modify: `tests/test_pc_config.py`, `tests/test_pc_flow_model.py`, `tests/test_pc_pipeline.py`, `tests/test_train_pc.py`

**Interfaces:**
- `make_pc_ddpm_batch(future_points, scheduler, generator) -> PCDDPMBatch`.
- `PCFlowModel(..., objective_type="flow" | "ddpm")`.
- `PCDDIMPipeline(model, scheduler)` and `create_pc_noise_scheduler(objective)`.

- [ ] **Step 1: Write failing behavior tests**

```python
def test_ddpm_batch_noises_absolute_positions_and_repeats_time():
    future = torch.full((2, 48, 1, 2, 3), 7.0)
    batch = make_pc_ddpm_batch(future, FakeDDPMScheduler(), torch.Generator().manual_seed(0))
    assert batch.target is future
    assert batch.model_input.shape == future.shape
    assert torch.equal(batch.frame_times, batch.timesteps[:, None].expand(-1, 49).to(future.dtype))


def test_ddpm_model_adds_source_to_zero_predicted_offset():
    model = make_tiny_model(objective_type="ddpm")
    torch.nn.init.zeros_(model.flow_head.projection.weight)
    torch.nn.init.zeros_(model.flow_head.projection.bias)
    source = torch.full((1, 1, 8, 3), 9.0)
    output = model(torch.zeros(1, 48, 1, 8, 3), torch.full((1, 49), 500.0), source, torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))
    assert torch.equal(output, source.unsqueeze(1).expand_as(output))


def test_ddpm_objective_creates_sample_prediction_scheduler():
    scheduler = create_pc_noise_scheduler({"type": "ddpm", "num_train_timesteps": 1000, "beta_schedule": "linear"})
    assert scheduler.config.prediction_type == "sample"
    assert scheduler.config.clip_sample is False
```

```python
def test_ddpm_model_rejects_unknown_objective():
    with pytest.raises(ValueError, match="objective_type"):
        make_tiny_model(objective_type="unknown")


def test_ddim_pipeline_does_not_add_source_after_sampling():
    output = PCDDIMPipeline(ZeroDenoiser(), FakeDDIMScheduler())(
        torch.full((1, 1, 2, 3), 7.0),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        "cpu",
        2,
    )

    assert torch.equal(output, torch.zeros_like(output))
```

- [ ] **Step 2: Verify red**

Run: `conda run -n das python -m pytest tests/test_pc_ddpm.py tests/test_pc_flow_model.py tests/test_pc_pipeline.py tests/test_train_pc.py -q`

Expected: missing DDPM units, model mode, pipeline, and scheduler helper failures.

- [ ] **Step 3: Implement the objective switch**

```python
@dataclass(frozen=True)
class PCDDPMBatch:
    model_input: torch.Tensor
    target: torch.Tensor
    frame_times: torch.Tensor
    timesteps: torch.Tensor


def make_pc_ddpm_batch(future_points, scheduler, generator):
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (future_points.shape[0],), device=future_points.device, generator=generator)
    noise = torch.randn(future_points.shape, device=future_points.device, dtype=future_points.dtype, generator=generator)
    return PCDDPMBatch(scheduler.add_noise(future_points, noise, timesteps), future_points, timesteps[:, None].expand(-1, 49).to(future_points.dtype), timesteps)
```

Validate `objective.type`, `num_train_timesteps`, `time_shift`, and linear DDPM beta schedule; set config type to DDPM. DDPM model mode embeds `[p0, noisy_absolute_positions]` and returns `head_output + p0`; retain flow model behavior. Add `PCDDIMPipeline` that samples absolute Gaussian states with DDIM and returns the scheduler result directly. Branch trainer batches and visualization between DDPM/DDIM and current flow/UniPC behavior.

- [ ] **Step 4: Verify green**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n das python -m pytest -q`

Expected: all tests pass; only existing Matplotlib/Torch deprecation warnings may remain.

- [ ] **Step 5: Commit**

```bash
git add configs/train/config_pc.yaml training/pc_config.py training/pc_ddpm.py wan/modules/pc_flow.py wan/pc_pipeline.py train_pc.py tests
git commit -m "feat: add selectable PC DDPM objective"
```
