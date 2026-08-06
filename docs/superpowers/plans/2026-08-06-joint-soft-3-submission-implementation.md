# Joint Soft 3 Submission Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a four-CPU SLURM launcher for the `td_832x480_3_soft` joint-training dataset with isolated `joint_soft_3` logs and checkpoints.

**Architecture:** Keep `submit_joint_3.sh` as the source template and add one independent sibling script. Extend the existing submission-script test module with a focused test that reads the new script and asserts its scheduler and trainer overrides.

**Tech Stack:** Bash, SLURM directives, pytest.

## Global Constraints

- Create `submit_joint_soft_3.sh`; do not modify `submit_joint_3.sh`.
- Use `#SBATCH --cpus-per-task=4`.
- Use exactly `joint_soft_3` for the SLURM job name and log/error filename prefix.
- Pass `data.dataset_root=td_832x480_3_soft`.
- Pass `logging.output_dir=outputs/joint_soft_3`.
- Preserve the H200, Singularity, Accelerate, resume, and max-step settings from `submit_joint_3.sh`.

---

### Task 1: Add the soft-dataset launcher and regression test

**Files:**
- Create: `submit_joint_soft_3.sh`
- Modify: `tests/test_submit_joint.py`

**Interfaces:**
- Consumes: the shared SLURM/Accelerate invocation pattern in `submit_joint_3.sh`.
- Produces: an executable submission script that trains against `td_832x480_3_soft` and resumes only from `outputs/joint_soft_3`.

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_submit_joint.py`:

```python
def test_soft_three_object_joint_launcher_uses_soft_dataset_and_isolated_output() -> None:
    script = Path("submit_joint_soft_3.sh").read_text()

    for expected in (
        "#SBATCH --job-name=joint_soft_3",
        "#SBATCH --cpus-per-task=4",
        "logs/joint_soft_3_%j.out",
        "logs/joint_soft_3_%j.err",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "data.dataset_root=td_832x480_3_soft",
        "logging.output_dir=outputs/joint_soft_3",
        "training.resume_from_checkpoint=latest",
        "training.max_train_steps=10000",
    ):
        assert expected in script
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
conda run -n das python -m pytest tests/test_submit_joint.py::test_soft_three_object_joint_launcher_uses_soft_dataset_and_isolated_output -q
```

Expected: failure because `submit_joint_soft_3.sh` does not exist yet.

- [ ] **Step 3: Create the minimal launcher**

Copy the complete contents of `submit_joint_3.sh` into `submit_joint_soft_3.sh`, then make these exact replacements:

```bash
#SBATCH --job-name=joint_soft_3
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_soft_3_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_soft_3_%j.err
        data.dataset_root=td_832x480_3_soft
        logging.output_dir=outputs/joint_soft_3
```

Keep every other directive, bind mount, launcher option, and trainer setting byte-for-byte equivalent to the source script.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
conda run -n das python -m pytest tests/test_submit_joint.py -q
```

Expected: all joint submission-script tests pass.

- [ ] **Step 5: Commit the focused implementation**

```bash
git add submit_joint_soft_3.sh tests/test_submit_joint.py
git commit -m "chore: add soft joint submission"
```

### Task 2: Verify the repository integration

**Files:**
- Verify: `submit_joint_soft_3.sh`
- Verify: `tests/test_submit_joint.py`

**Interfaces:**
- Consumes: the launcher and focused regression test from Task 1.
- Produces: a repository-verified submission script without changes to existing launchers.

- [ ] **Step 1: Inspect the scoped diff**

Run:

```bash
git diff --check
git diff -- submit_joint_soft_3.sh tests/test_submit_joint.py
```

Expected: only the new launcher and its focused test are changed, with no whitespace errors.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
conda run -n das python -m pytest -q
```

Expected: the complete pytest suite passes.

