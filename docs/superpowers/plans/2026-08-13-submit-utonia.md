# Utonia Slurm Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a requeueable A100 Slurm launcher for the Utonia-conditioned point-cloud overfit experiment.

**Architecture:** `submit_utonia.sh` mirrors the established joint-trajectory Slurm script so it inherits the same project path, Singularity image, bind mounts, diagnostics, and requeue behavior. It switches the GPU constraint, job/log identity, and Accelerate command to Utonia point-cloud training, with automatic latest-checkpoint resume.

**Tech Stack:** Slurm, Singularity, Accelerate, Bash, pytest.

## Global Constraints

- Use `gpu_requeue`, `--constraint=a100`, one GPU, four CPUs, 64 GB memory, a 10:30:00 wall time, `--requeue`, and `--open-mode=append`.
- Use `/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited` as `PROJECT_DIR` and `current.sif` from that directory.
- Retain `/n/holylabs`, `/net/holy-isilon`, and `/tmp:/dev/shm` Singularity binds.
- Launch `train_pc.py --config configs/train/config_pc_utonia_overfit.yaml training.resume_from_checkpoint=latest` through the H200 single-GPU Accelerate config.
- Use `utonia_%j.out` and `utonia_%j.err` log names under the project `logs/` directory.

---

## File Map

| File | Responsibility |
| --- | --- |
| `submit_utonia.sh` | Slurm submission script for Utonia PC overfit/requeue execution. |
| `tests/test_submit_utonia.py` | Static contract test for Slurm resources, container launch, and exact training command. |

## Tasks

### Task 1: Add the Utonia Slurm launcher

**Files:**
- Create: `submit_utonia.sh`
- Create: `tests/test_submit_utonia.py`

**Interfaces:**
- Consumes: the project checkout, `${PROJECT_DIR}/current.sif`, Slurm environment variables, and `configs/accelerate/h200_single_gpu.yaml`.
- Produces: an `sbatch submit_utonia.sh` entry point that resumes `config_pc_utonia_overfit.yaml` from its latest checkpoint.

- [ ] **Step 1: Write the failing static launcher test.**

```python
from pathlib import Path


def test_utonia_launcher_uses_h200_requeue_and_latest_resume():
    script = Path("submit_utonia.sh").read_text()

    for declaration in (
        "#SBATCH --job-name=utonia",
        "#SBATCH --partition=gpu_requeue",
        "#SBATCH --constraint=a100",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=64G",
        "#SBATCH --time=10:30:00",
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
        "logs/utonia_%j.out",
        "logs/utonia_%j.err",
    ):
        assert declaration in script

    assert '"${PROJECT_DIR}/current.sif"' in script
    assert "--config_file configs/accelerate/h200_single_gpu.yaml" in script
    assert "train_pc.py" in script
    assert "--config configs/train/config_pc_utonia_overfit.yaml" in script
    assert "training.resume_from_checkpoint=latest" in script
```

- [ ] **Step 2: Run the test to verify red.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_submit_utonia.py`

Expected: FAIL because `submit_utonia.sh` does not yet exist.

- [ ] **Step 3: Add `submit_utonia.sh`.** Use the full Slurm header, project-directory setup, `nvidia-smi` diagnostics, and Singularity binds from `submit_joint_trajectory_12_37.sh`, with this final command:

```bash
exec singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    -B /tmp:/dev/shm \
    "${PROJECT_DIR}/current.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_pc.py \
        --config configs/train/config_pc_utonia_overfit.yaml \
        training.resume_from_checkpoint=latest
```

- [ ] **Step 4: Run the static test to verify green.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_submit_utonia.py`

Expected: PASS.

- [ ] **Step 5: Run Bash syntax validation.**

Run: `bash -n submit_utonia.sh`

Expected: no output and exit code 0.

- [ ] **Step 6: Commit.**

```bash
git add submit_utonia.sh tests/test_submit_utonia.py
git commit -m "feat: add Utonia Slurm launcher"
```

### Task 2: Verify the integration boundary

**Files:**
- Verify only.

- [ ] **Step 1: Run the relevant launcher and configuration tests.**

Run: `conda run -n utonia-dev python -m pytest -q tests/test_submit_utonia.py tests/test_train_pc.py tests/test_current_def.py`

Expected: all tests pass.

- [ ] **Step 2: Check whitespace and worktree state.**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; unrelated local changes remain untouched.

- [ ] **Step 3: Report the submission command.**

Run: `sbatch submit_utonia.sh`

Expected: Slurm prints a submitted job ID. The script itself verifies node/GPU visibility before invoking the containerized trainer.
