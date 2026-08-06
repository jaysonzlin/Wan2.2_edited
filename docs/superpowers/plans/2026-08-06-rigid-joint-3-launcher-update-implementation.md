# Rigid Joint 3 Launcher Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the enabled rigid three-object joint objective at weight `0.001` with four CPU cores and unambiguous `joint_3_rigid_0.001` output naming.

**Architecture:** Modify the one existing SLURM launcher and add a focused content regression test in the established joint-launcher test module. The trainer itself is unchanged; the launcher supplies the necessary command-line overrides.

**Tech Stack:** Bash, SLURM directives, pytest.

## Global Constraints

- Modify only `submit_joint_3_rigid.sh` and `tests/test_submit_joint.py`.
- Use exactly `joint_3_rigid_0.001` for the job name and log/error filename prefix.
- Use `#SBATCH --cpus-per-task=4`.
- Use `logging.output_dir=outputs/joint_3_rigid_0.001`.
- Pass `objective.enable_rigid_loss=true`, `objective.rigid_loss_weight=0.001`, and `objective.rigid_loss_neighbors=4`.
- Preserve the existing H200, dataset `td_832x480_3`, Singularity, Accelerate, resume, and max-step settings.

---

### Task 1: Update and verify the rigid launcher

**Files:**
- Modify: `submit_joint_3_rigid.sh`
- Modify: `tests/test_submit_joint.py`

**Interfaces:**
- Consumes: `train_joint_wan_physctrl.py` command-line override syntax and the `objective.enable_rigid_loss` toggle.
- Produces: a scheduler submission command whose rigid objective actually runs with the requested coefficient and isolated checkpoints.

- [ ] **Step 1: Write the failing launcher-content test**

Append this test to `tests/test_submit_joint.py`:

```python
def test_rigid_three_object_joint_launcher_enables_the_configured_rigid_objective() -> None:
    script = Path("submit_joint_3_rigid.sh").read_text()

    for expected in (
        "#SBATCH --job-name=joint_3_rigid_0.001",
        "#SBATCH --cpus-per-task=4",
        "logs/joint_3_rigid_0.001_%j.out",
        "logs/joint_3_rigid_0.001_%j.err",
        "data.dataset_root=td_832x480_3",
        "logging.output_dir=outputs/joint_3_rigid_0.001",
        "objective.enable_rigid_loss=true",
        "objective.rigid_loss_weight=0.001",
        "objective.rigid_loss_neighbors=4",
    ):
        assert expected in script
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
conda run -n das python -m pytest tests/test_submit_joint.py::test_rigid_three_object_joint_launcher_enables_the_configured_rigid_objective -q
```

Expected: fail because the legacy launcher uses eight CPUs, the old names, a weight of `1.0`, and does not enable the toggle.

- [ ] **Step 3: Make the minimal launcher substitutions**

In `submit_joint_3_rigid.sh`, set:

```bash
#SBATCH --job-name=joint_3_rigid_0.001
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_3_rigid_0.001_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Wan2.2_edited/logs/joint_3_rigid_0.001_%j.err
        logging.output_dir=outputs/joint_3_rigid_0.001
        objective.enable_rigid_loss=true
        objective.rigid_loss_weight=0.001
        objective.rigid_loss_neighbors=4
```

Retain the existing `data.dataset_root=td_832x480_3` override and every unrelated launcher line.

- [ ] **Step 4: Run joint-launcher tests**

Run:

```bash
conda run -n das python -m pytest tests/test_submit_joint.py -q
```

Expected: all launcher tests pass.

- [ ] **Step 5: Commit the focused change**

```bash
git add submit_joint_3_rigid.sh tests/test_submit_joint.py
git commit -m "chore: tune rigid joint submission"
```

### Task 2: Verify repository integration

**Files:**
- Verify: `submit_joint_3_rigid.sh`
- Verify: `tests/test_submit_joint.py`

**Interfaces:**
- Consumes: the updated launcher and its focused test.
- Produces: a clean, repository-verified submission update.

- [ ] **Step 1: Review the scoped diff**

Run:

```bash
git diff --check
git diff -- submit_joint_3_rigid.sh tests/test_submit_joint.py
```

Expected: only the intended CPU, naming, output, and objective override changes are present.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
conda run -n das python -m pytest -q
```

Expected: the complete suite passes.

