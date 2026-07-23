# H200 Slurm Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the three existing I2V/TI2V job scripts to the H200 partition without changing their names or training commands.

**Architecture:** Existing per-script source-contract tests will assert the shared Slurm-header policy.  The three scripts will then receive only the approved directive changes.

**Tech Stack:** Bash, Slurm `#SBATCH` directives, pytest.

## Global Constraints

- Preserve script filenames, job names, output paths, commands, and training overrides.
- In all three scripts, set `#SBATCH --partition=gpu_h200`.
- Remove `#SBATCH --constraint=h200` and `#SBATCH --requeue` from all three scripts.

---

### Task 1: Enforce and apply the H200 Slurm header policy

**Files:**
- Modify: `submit_h200_i2v_requeue.sh:3-9`
- Modify: `submit_h200_i2v_lingbot_optim_requeue.sh:3-9`
- Modify: `submit_h200_ti2v_negative_requeue.sh:3-9`
- Modify: `tests/test_submit_i2v_requeue.py:4-11`
- Modify: `tests/test_submit_i2v_lingbot_optim_requeue.py:4-15`
- Modify: `tests/test_submit_ti2v_negative_requeue.py:4-12`

**Interfaces:**
- Consumes: the three existing Bash submission scripts.
- Produces: Slurm headers with the dedicated H200 partition and no requeue or H200 constraint directive.

- [ ] **Step 1: Write the failing header assertions**

  Add these assertions to the body of each existing submission-script test:

  ```python
      assert "#SBATCH --partition=gpu_h200" in script
      assert "#SBATCH --constraint=h200" not in script
      assert "#SBATCH --requeue" not in script
  ```

- [ ] **Step 2: Verify the tests fail before the script edits**

  Run:

  ```bash
  conda run -n das python -m pytest \
    tests/test_submit_i2v_requeue.py \
    tests/test_submit_i2v_lingbot_optim_requeue.py \
    tests/test_submit_ti2v_negative_requeue.py -v
  ```

  Expected: all three tests fail because their headers still request
  `gpu_requeue` and include the constraint and requeue directives.

- [ ] **Step 3: Apply the approved Slurm-header changes**

  In each script, replace:

  ```bash
  #SBATCH --partition=gpu_requeue
  #SBATCH --constraint=h200
  #SBATCH --requeue
  ```

  with:

  ```bash
  #SBATCH --partition=gpu_h200
  ```

  Leave every other line unchanged.

- [ ] **Step 4: Verify script syntax and policy tests**

  Run:

  ```bash
  bash -n submit_h200_i2v_requeue.sh \
    submit_h200_i2v_lingbot_optim_requeue.sh \
    submit_h200_ti2v_negative_requeue.sh
  conda run -n das python -m pytest \
    tests/test_submit_i2v_requeue.py \
    tests/test_submit_i2v_lingbot_optim_requeue.py \
    tests/test_submit_ti2v_negative_requeue.py -v
  ```

  Expected: all syntax checks succeed and all three tests pass.

- [ ] **Step 5: Commit the scripts and tests**

  ```bash
  git add \
    submit_h200_i2v_requeue.sh \
    submit_h200_i2v_lingbot_optim_requeue.sh \
    submit_h200_ti2v_negative_requeue.sh \
    tests/test_submit_i2v_requeue.py \
    tests/test_submit_i2v_lingbot_optim_requeue.py \
    tests/test_submit_ti2v_negative_requeue.py
  git commit -m "chore: target H200 Slurm partition"
  ```

### Task 2: Record the finalized submission policy

**Files:**
- Modify: `docs/superpowers/specs/2026-07-22-h200-slurm-submission-design.md`
- Modify: `docs/superpowers/plans/2026-07-22-h200-slurm-submission.md`

**Interfaces:**
- Consumes: the tested Slurm-header configuration.
- Produces: specification and plan documents describing the implemented H200-only submission policy.

- [ ] **Step 1: Verify documentation coverage**

  Confirm both documents state the three script names, `gpu_h200`, and removal
  of both `--constraint=h200` and `--requeue` without filename or output-path
  changes.

- [ ] **Step 2: Inspect the final working tree**

  Run:

  ```bash
  git diff --check HEAD
  git status --short
  ```

  Expected: no whitespace errors; only the plan document and documented
  pre-existing local files remain before the documentation commit.

- [ ] **Step 3: Commit the plan document**

  ```bash
  git add docs/superpowers/plans/2026-07-22-h200-slurm-submission.md
  git commit -m "docs: plan H200 Slurm submission update"
  ```
