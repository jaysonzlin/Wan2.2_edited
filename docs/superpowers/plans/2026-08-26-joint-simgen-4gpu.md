# Joint SimGen 4-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit four-H200-GPU Accelerate workflow for Joint SimGen training without changing the existing single-GPU workflow.

**Architecture:** Accelerate remains responsible for DDP wrapping and training-data sharding. Small helpers make cache launch safety, shared RNG construction, and training metric reduction testable; rank zero evaluates the fixed validation split without padding; profile files and a Slurm entry point carry deployment details.

**Tech Stack:** Python 3, PyTorch, Hugging Face Accelerate >=1.1.1, pytest, YAML, Slurm, Singularity.

**Spec:** `docs/superpowers/specs/2026-08-26-joint-simgen-4gpu-design.md`

## Global Constraints

- Preserve the existing single-GPU YAML files unchanged.
- Use one H200 node: `gpu_requeue`, `--constraint=h200`, four GPUs, 16 CPUs, 128 GB, 12 hours, and requeue enabled.
- Retain local batch size one and set the four-GPU profile to 10,000 optimizer steps.
- All ranks use the configured base seed, with no rank offset.
- Reject multi-process cache creation; normal training only reads the cache.
- Cross-world-size resumes retain Accelerate-loadable shared state and reset every rank's RNG to the configured base seed.
- Validation must be the unpadded mean over the unique ten-example validation split on rank zero.

---

### Task 1: Distributed-safe Joint SimGen runtime

**Files:**
- Modify: `joint_simgen.py:1-381`
- Modify: `tests/test_joint_simgen.py`

**Interfaces:**
- Produces: `_cache_preparation_world_size() -> int`, `_shared_generator(device, seed) -> torch.Generator`, and `_reduced_mean(accelerator, local_sum, local_count) -> torch.Tensor`.
- Consumes: `Accelerator` and the existing checkpoint helpers.

- [ ] **Step 1: Write failing tests**

```python
def test_main_rejects_distributed_cache_preparation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(joint_simgen, "load_simgen_joint_config", lambda *_: {"data": {}})
    monkeypatch.setattr("sys.argv", ["joint_simgen.py", "--config", "config.yaml", "--prepare-utonia-cache"])
    with pytest.raises(ValueError, match="single GPU"):
        joint_simgen.main()

def test_shared_generator_restarts_from_configured_seed():
    assert torch.equal(
        torch.rand(4, generator=joint_simgen._shared_generator("cpu", 42)),
        torch.rand(4, generator=joint_simgen._shared_generator("cpu", 42)),
    )

def test_reduced_mean_uses_global_loss_sum_and_example_count():
    accelerator = SumReducer([torch.tensor(5.0), torch.tensor(2)])
    assert joint_simgen._reduced_mean(accelerator, torch.tensor(5.0), 2).item() == 2.5
```

The third test catches a regression that averages per-rank means rather than weighting by the number of unique examples.

- [ ] **Step 2: Verify the red state**

Run: `pytest tests/test_joint_simgen.py -q`

Expected: cache preparation is not rejected and helper symbols are absent.

- [ ] **Step 3: Implement the helpers and use them**

```python
def _cache_preparation_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def _shared_generator(device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)

def _reduced_mean(accelerator, local_sum: torch.Tensor, local_count: int) -> torch.Tensor:
    return accelerator.reduce(local_sum, reduction="sum") / accelerator.reduce(
        local_sum.new_tensor(local_count), reduction="sum"
    )
```

Reject cache mode before its writer for `WORLD_SIZE > 1`. Keep Accelerate's default even training batches; call `set_seed(seed, device_specific=False)` after resume and rebuild the diffusion generator with the same seed. Reduce train scalars before rank-zero logging. Keep the validation loader out of `accelerator.prepare` and evaluate its ten unpadded examples only on rank zero with the unwrapped model. Use the unwrapped model for the rank-zero visualization loss forward and synchronize around main-process filesystem work.

- [ ] **Step 4: Verify the green state**

Run: `pytest tests/test_joint_simgen.py -q`

Expected: all Joint SimGen tests pass.

- [ ] **Step 5: Commit**

```bash
git add joint_simgen.py tests/test_joint_simgen.py
git commit -m "feat: make Joint SimGen distributed-safe"
```

### Task 2: Four-GPU profile and H200 requeue launcher

**Files:**
- Create: `configs/accelerate/h200_4gpu.yaml`
- Create: `configs/train/joint_simgen_480_4gpu.yaml`
- Create: `submit_joint_simgen_4gpu.sh`
- Modify: `tests/test_joint_simgen.py`

**Interfaces:**
- Produces: `accelerate launch --config_file configs/accelerate/h200_4gpu.yaml joint_simgen.py --config configs/train/joint_simgen_480_4gpu.yaml`.

- [ ] **Step 1: Write a failing profile test**

```python
def test_4gpu_profile_has_four_processes_and_separate_output():
    accelerate_config = yaml.safe_load(Path("configs/accelerate/h200_4gpu.yaml").read_text())
    training_config = yaml.safe_load(Path("configs/train/joint_simgen_480_4gpu.yaml").read_text())
    assert accelerate_config["distributed_type"] == "MULTI_GPU"
    assert accelerate_config["num_processes"] == 4
    assert training_config["training"]["max_train_steps"] == 10_000
    assert training_config["logging"]["output_dir"] == "outputs/joint_simgen_4gpu"
```

- [ ] **Step 2: Verify the red state**

Run: `pytest tests/test_joint_simgen.py::test_4gpu_profile_has_four_processes_and_separate_output -q`

Expected: FAIL because the profiles do not exist.

- [ ] **Step 3: Create the profile artifacts**

Copy all existing single-GPU Accelerate fields but set `distributed_type: MULTI_GPU` and `num_processes: 4`. Copy the Joint SimGen training YAML but set only `training.max_train_steps: 10000` and `logging.output_dir: outputs/joint_simgen_4gpu`. Add a Slurm script retaining existing Singularity binds and launching those two files with `gpu_requeue`, H200 constraint, four GPUs, 16 CPUs, 128 GB, 12 hours, and requeue enabled.

- [ ] **Step 4: Verify the green state**

Run: `pytest tests/test_joint_simgen.py::test_4gpu_profile_has_four_processes_and_separate_output -q && bash -n submit_joint_simgen_4gpu.sh`

Expected: pytest passes and Bash exits zero.

- [ ] **Step 5: Commit**

```bash
git add configs/accelerate/h200_4gpu.yaml configs/train/joint_simgen_480_4gpu.yaml submit_joint_simgen_4gpu.sh tests/test_joint_simgen.py
git commit -m "feat: add Joint SimGen four-GPU launch profile"
```

### Task 3: Document and verify the workflow

**Files:**
- Modify: `README.md:3-18`
- Modify: `tests/test_joint_simgen.py`

**Interfaces:**
- Consumes: the paths created in Task 2 and the cache guard from Task 1.
- Produces: copyable cache-first and four-GPU commands.

- [ ] **Step 1: Write a failing documentation test**

```python
def test_readme_documents_single_gpu_cache_then_four_gpu_training():
    readme = Path("README.md").read_text()
    assert "h200_4gpu.yaml" in readme
    assert "joint_simgen_480_4gpu.yaml" in readme
    assert "single-GPU" in readme
```

- [ ] **Step 2: Verify the red state**

Run: `pytest tests/test_joint_simgen.py::test_readme_documents_single_gpu_cache_then_four_gpu_training -q`

Expected: FAIL because the new profile names are absent.

- [ ] **Step 3: Update README**

Keep the cache command on the single-GPU config. Add the four-GPU launch and `sbatch submit_joint_simgen_4gpu.sh` commands, plus the fixed-ten-example validation and shared-seed resume behavior.

- [ ] **Step 4: Verify local behavior**

Run: `pytest tests/test_joint_simgen.py tests/test_simgen_joint_config.py -q && python -m py_compile joint_simgen.py && bash -n submit_joint_simgen_4gpu.sh && git diff --check`

Expected: every command exits zero.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_joint_simgen.py
git commit -m "docs: document Joint SimGen four-GPU workflow"
```

### Task 4: H200 operational smoke test

**Files:**
- Verify only.

- [ ] **Step 1: Run the cache command once on one GPU**

```bash
accelerate launch --config_file configs/accelerate/h200_single_gpu.yaml \
  joint_simgen.py --config configs/train/joint_simgen_480.yaml --prepare-utonia-cache
```

- [ ] **Step 2: Submit the four-GPU training job**

```bash
sbatch submit_joint_simgen_4gpu.sh
```

Inspect the log for four local ranks, rank-zero logging, checkpoint creation, and a validation loss at step 250. Resume the output once from the four-GPU profile and once from the single-GPU profile; both must restore the available model/optimizer/scheduler/step state and reset rank RNGs from `training.seed`.
