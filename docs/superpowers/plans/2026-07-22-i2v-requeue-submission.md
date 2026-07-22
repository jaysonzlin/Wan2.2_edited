# I2V Requeue Submission Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an H200 requeue Slurm script that runs image-conditioned I2V training with an empty text prompt and an isolated output directory.

**Architecture:** Copy the existing I2V requeue submission settings into a separately named shell script. Keep the execution environment and scheduler settings identical while changing only the job/log identity and training overrides needed for an empty-prompt I2V run.

**Tech Stack:** Bash, Slurm, Singularity, Accelerate, pytest.

## Global Constraints

- The entrypoint is `train_i2v.py`; do not invoke point-cloud training.
- Override `data.prompt=""` so all samples use the empty text condition.
- Override `logging.output_dir=outputs/i2v_requeue` so checkpoints, visualizations, and resume lookup remain isolated.
- Preserve the H200 requeue allocation, Singularity image, Accelerate config, checkpoint cadence, and constant LR override from `submit_h200_requeue.sh`.

---

### Task 1: Add the empty-prompt I2V submission script

**Files:**

- Create: `submit_h200_i2v_requeue.sh`
- Create: `tests/test_submit_i2v_requeue.py`

**Interfaces:**

- Consumes: the Slurm, Singularity, Accelerate, and `train_i2v.py` invocation conventions in `submit_h200_requeue.sh`.
- Produces: an executable-compatible Bash script whose command line targets empty-prompt I2V training in `outputs/i2v_requeue`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_submit_i2v_requeue.py`:

```python
from pathlib import Path


def test_i2v_requeue_script_uses_empty_prompt_and_isolated_output() -> None:
    script = Path("submit_h200_i2v_requeue.sh").read_text()

    assert "#SBATCH --job-name=wan_i2v" in script
    assert "train_i2v.py" in script
    assert 'data.prompt=""' in script
    assert "logging.output_dir=outputs/i2v_requeue" in script
    assert "train_pc.py" not in script
```

- [ ] **Step 2: Run the test to verify it fails**

Run `conda run -n das python -m pytest tests/test_submit_i2v_requeue.py -v`.

Expected: FAIL with `FileNotFoundError` because the new script does not exist.

- [ ] **Step 3: Create the script**

Copy `submit_h200_requeue.sh` to `submit_h200_i2v_requeue.sh`. Change the Slurm identity to `wan_i2v`:

```bash
#SBATCH --job-name=wan_i2v
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_i2v_%j.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/jaysonzlin/Wan2.2_edited/logs/wan_i2v_%j.err
```

Preserve the existing `accelerate launch ... train_i2v.py` command and add these OmegaConf overrides:

```bash
        data.prompt="" \
        logging.output_dir=outputs/i2v_requeue \
```

Place them after `--config configs/train/overfit_kubric_i2v.yaml` and before the existing training overrides.

- [ ] **Step 4: Verify syntax and the focused test**

Run `bash -n submit_h200_i2v_requeue.sh && conda run -n das python -m pytest tests/test_submit_i2v_requeue.py -v`.

Expected: Bash syntax check succeeds and the test passes.

- [ ] **Step 5: Commit the submission variant**

Run `git add submit_h200_i2v_requeue.sh tests/test_submit_i2v_requeue.py` and `git commit -m "feat: add empty-prompt I2V requeue script"`.

## Final verification

- [ ] Run `bash -n submit_h200_i2v_requeue.sh`.
- [ ] Run `conda run -n das python -m pytest tests/test_submit_i2v_requeue.py -v`.
- [ ] Run `git diff --check` and confirm there are no whitespace errors.
