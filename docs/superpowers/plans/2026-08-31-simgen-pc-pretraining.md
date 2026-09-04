# SimGen PC Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable, four-GPU SimGen point-cloud pretraining program that exports the best joint-compatible PC-model weights.

**Architecture:** `training/simgen_pc_pretraining.py` is the deep module at the SimGen-to-PC seam. It flattens variable-object samples, builds the history-DDPM PC objective, reduces validation loss by object count, saves the best PC state, and renders sample 0. `pretrain_simgen_pc.py` only orchestrates Accelerate, datasets, optimizer, checkpoints, and calls to that module.

**Tech Stack:** Python 3.10, PyTorch, Accelerate, Diffusers, YAML, pytest, Slurm, Singularity.

**Spec:** `docs/superpowers/specs/2026-08-31-simgen-pc-pretraining-design.md`

## Global Constraints

- Use read-only `SimGenJointDataset` and `SimGenUtoniaCache`; do not create Utonia extractors, Wan, or bridge modules.
- Match the joint PC model: 2,048 points; 49 frames; four history frames; 45 future frames; width 256; eight layers; four heads; Utonia conditioning; DDPM x0 prediction.
- Train samples 0–127 and validate samples 490–499. One data-loader item is one native sample; every object becomes one PC-model item.
- Use four GPUs on one node, 200,000 optimizer steps, and a 1,000-step checkpoint, validation, and visualization cadence.
- Retain exactly two numeric `checkpoint-*` directories. `best_pc_model.pt` and `best_pc_model.json` are separate and never pruned.
- Export a plain `PCTrajectoryModel.state_dict()` accepted strictly by `joint_simgen.load_pretrained_pc_weights`.

---

### Task 1: Create the SimGen PC pretraining module

**Files:**
- Create: `training/simgen_pc_pretraining.py`
- Test: `tests/test_simgen_pc_pretraining.py`

**Interfaces:**
- Consumes: `simgen_joint_collate` batches containing `point_clouds [1, K, 49, 1, 2048, 3]` and `utonia_features [1, K, 2048, D]`, plus `make_pc_ddpm_batch`.
- Produces: `SimGenPCBatch`, `flatten_simgen_pc_batch`, `pc_pretraining_prediction`, `object_mse_totals`, `reduce_object_mean`, `load_best_export_metadata`, `save_best_pc_export`, and `save_sample_zero_visualizations`.

- [ ] **Step 1: Write the failing module tests**

```python
class RecordingModel:
    def __call__(self, noisy_future, frame_times, history, *_unused, utonia_features):
        self.history = history
        self.features = utonia_features
        return torch.zeros_like(noisy_future)


class SumReducer:
    def __init__(self, values):
        self.values = iter(values)

    def reduce(self, _value, reduction):
        assert reduction == "sum"
        return next(self.values)


def make_two_object_simgen_batch():
    return {
        "point_clouds": torch.zeros(1, 2, 49, 1, 2048, 3),
        "utonia_features": torch.zeros(1, 2, 2048, 5),
    }


def test_flatten_simgen_objects_preserves_each_history_future_and_feature():
    batch = make_two_object_simgen_batch()

    flattened = flatten_simgen_pc_batch(batch, device="cpu")

    assert flattened.history.shape == (2, 4, 1, 2048, 3)
    assert flattened.future.shape == (2, 45, 1, 2048, 3)
    assert torch.equal(flattened.history[1], batch["point_clouds"][0, 1, :4])
    assert torch.equal(flattened.utonia_features[0], batch["utonia_features"][0, 0])


def test_pc_pretraining_prediction_uses_history_and_clean_future_target():
    recording_model = RecordingModel()
    flattened_batch = flatten_simgen_pc_batch(make_two_object_simgen_batch(), "cpu")
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    generator = torch.Generator().manual_seed(0)
    prediction, target = pc_pretraining_prediction(
        flattened_batch, recording_model, ddpm_scheduler, generator, "cpu"
    )

    assert prediction.shape == target.shape == (2, 45, 1, 2048, 3)
    assert torch.equal(target, flattened_batch.future)
    assert torch.equal(recording_model.history, flattened_batch.history)
    assert torch.equal(recording_model.features, flattened_batch.utonia_features)


def test_object_weighted_reduction_uses_global_object_count():
    accelerator = SumReducer([torch.tensor(12.0), torch.tensor(3.0)])

    assert reduce_object_mean(accelerator, torch.tensor(4.0), 1).item() == 4.0
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `pytest tests/test_simgen_pc_pretraining.py -q`

Expected: FAIL during import because `training.simgen_pc_pretraining` does not exist.

- [ ] **Step 3: Implement the minimal deep module**

```python
@dataclass(frozen=True)
class SimGenPCBatch:
    history: torch.Tensor
    future: torch.Tensor
    utonia_features: torch.Tensor


def flatten_simgen_pc_batch(batch, device) -> SimGenPCBatch:
    points = batch["point_clouds"].to(device, non_blocking=True)
    features = batch["utonia_features"].to(device, non_blocking=True)
    if points.shape[0] != 1 or points.shape[2:] != (49, 1, 2048, 3):
        raise ValueError("expected one SimGen sample with [K, 49, 1, 2048, 3] point clouds")
    return SimGenPCBatch(points[0, :, :4], points[0, :, 4:], features[0])


def pc_pretraining_prediction(batch, model, scheduler, generator, device):
    ddpm = make_pc_ddpm_batch(batch.future, scheduler, generator, known_frames=4)
    return model(
        ddpm.model_input, ddpm.frame_times, batch.history, None, None,
        utonia_features=batch.utonia_features,
    ), ddpm.target
```

Implement `object_mse_totals` as a per-object MSE mean over every dimension after object, returning its sum and object count. `reduce_object_mean` must reduce sum and count separately with `accelerator.reduce(..., reduction="sum")`.

- [ ] **Step 4: Add failing persistence and visualization tests**

```python
def test_best_export_replaces_only_a_lower_validation_loss(tmp_path):
    model = torch.nn.Linear(2, 1)
    assert save_best_pc_export(model, tmp_path, validation_loss=0.4, step=100, best_loss=float("inf"))
    assert not save_best_pc_export(model, tmp_path, validation_loss=0.5, step=200, best_loss=0.4)

    assert load_best_export_metadata(tmp_path) == (0.4, 100)
    assert (tmp_path / "best_pc_model.pt").is_file()


def test_sample_zero_visualization_writes_one_path_per_object(monkeypatch, tmp_path):
    class FakeHistoryPipeline:
        def __call__(self, history, _device, _steps, _generator, *, utonia_features):
            return torch.zeros(history.shape[0], 45, 1, 2048, 3)

    monkeypatch.setattr(
        "training.simgen_pc_pretraining.save_pointcloud_comparison_mp4",
        lambda _prediction, _ground_truth, path, _fps: Path(path).touch(),
    )
    batch = flatten_simgen_pc_batch(make_two_object_simgen_batch(), "cpu")
    save_sample_zero_visualizations(FakeHistoryPipeline(), batch, tmp_path, 1000, fps=12)

    assert (tmp_path / "visualizations" / "step_0001000" / "object_000_trajectory_comparison.mp4").is_file()
    assert (tmp_path / "visualizations" / "step_0001000" / "object_001_trajectory_comparison.mp4").is_file()
```

- [ ] **Step 5: Implement persistence and rendering**

`load_best_export_metadata` returns `(float("inf"), 0)` when no metadata exists, otherwise validates JSON `{"validation_loss": float, "step": int}`. `save_best_pc_export` saves `model.state_dict()` and that JSON only when the new loss is lower; write each to a same-directory temporary path and atomically replace the destination.

Use `PCHistoryDDIMPipeline` with a DDIM scheduler matching the linear training scheduler. Under `torch.no_grad()`, sample the four-frame history and cached Utonia features, concatenate history with predicted future, and call `save_pointcloud_comparison_mp4` once per object under `visualizations/step_{step:07d}/`.

- [ ] **Step 6: Run the module tests and confirm green**

Run: `pytest tests/test_simgen_pc_pretraining.py -q`

Expected: PASS; it proves object flattening, history/DDPM target alignment, object-weighted reduction, best-export replacement, and one sample-0 artifact per object.

- [ ] **Step 7: Commit Task 1**

```bash
git add training/simgen_pc_pretraining.py tests/test_simgen_pc_pretraining.py
git commit -m "feat: add SimGen PC pretraining module"
```

### Task 2: Add the fixed pretraining configuration

**Files:**
- Create: `training/simgen_pc_pretraining_config.py`
- Create: `configs/train/pretrain_simgen_pc_480_4gpu.yaml`
- Test: `tests/test_simgen_pc_pretraining_config.py`

**Interfaces:**
- Consumes: a YAML path and dotted overrides.
- Produces: `load_simgen_pc_pretraining_config(path, overrides) -> dict`.

- [ ] **Step 1: Write the failing configuration tests**

```python
def test_pretraining_config_loads_the_fixed_200k_experiment():
    config = load_simgen_pc_pretraining_config(
        "configs/train/pretrain_simgen_pc_480_4gpu.yaml", []
    )

    assert config["data"]["train_end"] == 127
    assert config["training"]["max_train_steps"] == 200_000
    assert config["training"]["checkpoints_total_limit"] == 2
    assert config["validation"]["every_steps"] == 1000


def test_pretraining_config_rejects_non_joint_compatible_history(tmp_path):
    path = tmp_path / "invalid.yaml"
    config = yaml.safe_load(Path("configs/train/pretrain_simgen_pc_480_4gpu.yaml").read_text())
    config["model"]["history_frames"] = 1
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="history_frames must be 4"):
        load_simgen_pc_pretraining_config(path, [])
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `pytest tests/test_simgen_pc_pretraining_config.py -q`

Expected: FAIL during import because the dedicated config loader does not exist.

- [ ] **Step 3: Implement the validator and YAML**

Use `training.pc_config._apply_override`. Validate: 480×480, 49 frames, 2,048 points, samples 0–127/490–499, batch size one, history frames four, history/DDPM x0 model contract, 200,000 maximum steps, a 1,000-step cadence, and checkpoint limit two.

Set AdamW to the established joint PC settings: `lr: 1.0e-4`, `betas: [0.9, 0.999]`, `eps: 1.0e-8`, and `weight_decay: 0.01`. Set `training.resume_from_checkpoint: null` in YAML; the launcher supplies `latest` for requeue-safe execution.

- [ ] **Step 4: Run the configuration tests and confirm green**

Run: `pytest tests/test_simgen_pc_pretraining_config.py -q`

Expected: PASS; valid shipped config succeeds and bad contract overrides fail before training setup.

- [ ] **Step 5: Commit Task 2**

```bash
git add training/simgen_pc_pretraining_config.py configs/train/pretrain_simgen_pc_480_4gpu.yaml tests/test_simgen_pc_pretraining_config.py
git commit -m "feat: configure SimGen PC pretraining"
```

### Task 3: Implement Accelerate orchestration, recovery, and launcher

**Files:**
- Create: `pretrain_simgen_pc.py`
- Create: `submit_pretrain_simgen_pc_4gpu.sh`
- Modify: `tests/test_simgen_pc_pretraining.py`
- Modify: `tests/test_joint_simgen.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1/2 interfaces, `SimGenJointDataset`, `PCTrajectoryModel`, `train_pc.load_pc_checkpoint_with_fallback`, `train_pc.prune_pc_checkpoints`, and `configs/accelerate/h200_4gpu.yaml`.
- Produces: a CLI `pretrain_simgen_pc.py --config PATH [section.key=value ...]`, recoverable `checkpoint-<step>` directories, `best_pc_model.pt`, and the four-GPU Slurm command.

- [ ] **Step 1: Write failing recovery, transfer, and launcher tests**

```python
def test_pretraining_latest_resume_falls_back_from_an_incomplete_checkpoint(tmp_path):
    class FailingNewestAccelerator:
        def __init__(self):
            self.attempts = []

        def load_state(self, checkpoint):
            self.attempts.append(Path(checkpoint).name)
            if Path(checkpoint).name == "checkpoint-2000":
                raise RuntimeError("incomplete")

    for step in (1000, 2000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    accelerator = FailingNewestAccelerator()

    restored = load_pc_checkpoint_with_fallback(accelerator, tmp_path, "latest")

    assert restored.name == "checkpoint-1000"
    assert accelerator.attempts == ["checkpoint-2000", "checkpoint-1000"]


def test_joint_loader_accepts_a_best_pretraining_export(tmp_path):
    source_pc_model = torch.nn.Linear(2, 1)
    target_pc_model = torch.nn.Linear(2, 1)
    torch.save(source_pc_model.state_dict(), tmp_path / "best_pc_model.pt")

    joint_simgen.load_pretrained_pc_weights(target_pc_model, tmp_path / "best_pc_model.pt")

    assert all(
        torch.equal(source, target)
        for source, target in zip(source_pc_model.state_dict().values(), target_pc_model.state_dict().values())
    )


def test_pretraining_launcher_uses_four_gpus_and_latest_resume():
    source = Path("submit_pretrain_simgen_pc_4gpu.sh").read_text()
    assert "configs/accelerate/h200_4gpu.yaml" in source
    assert "pretrain_simgen_pc.py" in source
    assert "training.resume_from_checkpoint=latest" in source
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `pytest tests/test_simgen_pc_pretraining.py tests/test_joint_simgen.py -q`

Expected: FAIL because the pretraining entry point and launcher are absent.

- [ ] **Step 3: Implement the training loop**

Copy the ordering in `joint_simgen.run_training`: instantiate `SimGenUtoniaCache` before datasets and optimizer; create the fixed train/validation datasets; obtain cache width from `simgen_joint_collate([train_dataset[0]])`; build the exact PC model; create AdamW and scheduler; and call `accelerator.prepare(model, optimizer, train_loader, validation_loader, scheduler)`.

All ranks call `load_pc_checkpoint_with_fallback` after preparation. Restore best metadata after this call. Each rank trains with Task 1's PC objective. At each 1,000-step cadence all ranks save state; rank zero calls `prune_pc_checkpoints(output_dir, 2)`; then every rank waits. At validation cadence, all ranks evaluate their prepared validation shard under `torch.no_grad()`, reduce Task 1 totals by total object count, and rank zero conditionally writes the best export. Rank zero then renders sample 0 and all ranks wait before returning the model to training mode.

Copy Slurm, Singularity bindings, requeue configuration, and GPU diagnostics from `submit_joint_simgen_4gpu.sh`; change only names, entry point, config, and final override to `training.resume_from_checkpoint=latest`.

- [ ] **Step 4: Document the operator handoff**

Add a README pretraining section with the launcher command, the two-checkpoint retention/fallback behavior, the `best_pc_model.pt` path, and this fresh-joint invocation:

```bash
training.pretrained_pc_weights=/absolute/path/best_pc_model.pt \
training.resume_from_checkpoint=null
```

State that this must not be combined with a joint resume.

- [ ] **Step 5: Run the complete relevant suite and confirm green**

Run: `pytest tests/test_simgen_pc_pretraining.py tests/test_simgen_pc_pretraining_config.py tests/test_joint_simgen.py tests/test_train_pc.py tests/test_simgen_joint_dataset.py -q`

Expected: PASS with no failures.

- [ ] **Step 6: Commit Task 3**

```bash
git add pretrain_simgen_pc.py submit_pretrain_simgen_pc_4gpu.sh README.md tests/test_simgen_pc_pretraining.py tests/test_joint_simgen.py
git commit -m "feat: add resumable SimGen PC pretraining"
```
