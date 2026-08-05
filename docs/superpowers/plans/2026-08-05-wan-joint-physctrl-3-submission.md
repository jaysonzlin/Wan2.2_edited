# Three-Object Joint Wan–PhysCtrl Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Slurm launcher for joint Wan–PhysCtrl training on the cluster-local `td_832x480_3` dataset.

**Architecture:** Create `submit_joint_3.sh` by copying the existing joint launcher and changing only job-specific identifiers and dotted configuration overrides. Extend the joint launcher test to verify that the new script preserves the H200 launch contract while selecting the three-object dataset and separate output directory.

**Tech Stack:** Bash, Slurm, Singularity, Accelerate, pytest.

## Global Constraints

- New script filename: `submit_joint_3.sh`.
- Slurm job name and log prefix: `wan_joint_physctrl_3`.
- Dataset override: `data.dataset_root=td_832x480_3`.
- Training output override: `logging.output_dir=outputs/joint_wan_physctrl_3`.
- Keep the existing joint launcher’s resources, container invocation, config, checkpoint resume, and 10,000-step override unchanged.

---

### Task 1: Add the three-object joint submission launcher

**Files:**

- Create: `submit_joint_3.sh`
- Modify: `tests/test_submit_joint.py`

**Interfaces:**

- Consumes: `submit_joint.sh` as the canonical joint Slurm/Accelerate launch contract.
- Produces: an executable-compatible Bash submission script that starts `train_joint_wan_physctrl.py` with cluster-local three-object data and an isolated run directory.

- [ ] **Step 1: Write the failing launcher-contract test**

Append this test to `tests/test_submit_joint.py`:

```python
def test_three_object_joint_launcher_uses_isolated_dataset_and_output() -> None:
    script = Path("submit_joint_3.sh").read_text()

    for expected in (
        "#SBATCH --job-name=wan_joint_physctrl_3",
        "logs/wan_joint_physctrl_3_%j.out",
        "logs/wan_joint_physctrl_3_%j.err",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "data.dataset_root=td_832x480_3",
        "logging.output_dir=outputs/joint_wan_physctrl_3",
        "training.resume_from_checkpoint=latest",
        "training.max_train_steps=10000",
    ):
        assert expected in script
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/test_submit_joint.py -q`

Expected: FAIL with `FileNotFoundError` because `submit_joint_3.sh` does not exist.

- [ ] **Step 3: Create the derived launcher**

Copy `submit_joint.sh` to `submit_joint_3.sh`, retaining all existing directives and commands. Replace the job name and both log prefixes with `wan_joint_physctrl_3`. Add these training overrides after the existing config argument:

```bash
data.dataset_root=td_832x480_3 \
logging.output_dir=outputs/joint_wan_physctrl_3 \
```

Keep the pre-existing `training.resume_from_checkpoint=latest` and `training.max_train_steps=10000` overrides after them.

- [ ] **Step 4: Verify the launcher**

Run: `bash -n submit_joint_3.sh && pytest tests/test_submit_joint.py -q`

Expected: Bash exits 0 and both joint launcher tests pass.

- [ ] **Step 5: Commit**

```bash
git add submit_joint_3.sh tests/test_submit_joint.py
git commit -m "chore: add three-object joint submission"
```
