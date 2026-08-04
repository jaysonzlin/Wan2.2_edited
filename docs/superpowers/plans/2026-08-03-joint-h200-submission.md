# Joint Wan PhysCtrl H200 Submission Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Slurm script that launches the joint Wan PhysCtrl trainer on one H200 GPU for eight hours and resumes the newest checkpoint.

**Architecture:** The script is a single, focused Slurm entry point. It follows the repository’s existing H200 scripts: establish the shared project directory, emit job diagnostics, then replace the shell with Singularity running `accelerate` and the joint trainer.

**Tech Stack:** Bash, Slurm, Singularity, Hugging Face Accelerate.

## Global Constraints

- Partition: `gpu_h200`; resources: one GPU, eight CPUs, 64 GB memory, and `08:00:00` wall time.
- Launcher config: `configs/accelerate/h200_single_gpu.yaml`.
- Trainer and config: `train_joint_wan_physctrl.py` and `configs/train/joint_wan_physctrl_832x480.yaml`.
- Always append `training.resume_from_checkpoint=latest` without modifying the YAML.
- Use the existing shared project path, `current.sif` image, and `/n/holylabs`, `/net/holy-isilon`, and `/tmp:/dev/shm` bind mounts.

---

### Task 1: Add and validate the joint H200 submission entry point

**Files:**
- Create: `submit_joint.sh`
- Create: `tests/test_submit_joint.py`

**Interfaces:**
- Consumes: Slurm environment variables, the project container image, Accelerate’s H200 configuration, and the joint training YAML.
- Produces: An `sbatch submit_joint.sh`-compatible job that resumes the latest joint-training checkpoint.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_submit_joint_requests_h200_and_launches_joint_trainer() -> None:
    script = Path("submit_joint.sh").read_text()

    for expected in (
        "#SBATCH --partition=gpu_h200",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=08:00:00",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "training.resume_from_checkpoint=latest",
    ):
        assert expected in script
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_submit_joint.py -v`

Expected: FAIL because `submit_joint.sh` does not yet exist.

- [ ] **Step 3: Write the minimal implementation**

```bash
#!/bin/bash
#SBATCH --job-name=wan_joint_physctrl
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited"
cd "${PROJECT_DIR}"
mkdir -p logs

exec singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    -B /tmp:/dev/shm \
    "${PROJECT_DIR}/current.sif" \
    accelerate launch \
        --config_file configs/accelerate/h200_single_gpu.yaml \
        train_joint_wan_physctrl.py \
        --config configs/train/joint_wan_physctrl_832x480.yaml \
        training.resume_from_checkpoint=latest
```

Add dedicated Slurm output and error directives under the time directive that
write `wan_joint_physctrl_%j.out` and `wan_joint_physctrl_%j.err` in
`${PROJECT_DIR}/logs`. Before `exec`, print the job ID, node, time, CUDA device
selection, and `nvidia-smi`, as used by existing scripts.

- [ ] **Step 4: Run the focused checks**

Run: `pytest tests/test_submit_joint.py -v && bash -n submit_joint.sh`

Expected: pytest reports one passing test and Bash exits with status 0.

- [ ] **Step 5: Commit**

```bash
git add submit_joint.sh tests/test_submit_joint.py
git commit -m "chore: add joint H200 training submission script"
```
