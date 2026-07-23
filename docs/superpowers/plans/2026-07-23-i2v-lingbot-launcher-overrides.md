# I2V LingBot Launcher Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit the LingBot-optimizer I2V job with learning rate `1.0e-6` and write its artifacts to `outputs/i2v_lingbot_optim`.

**Architecture:** Make two command-line override changes in the existing LingBot I2V Slurm launcher. The launcher remains the sole experiment-specific configuration layer, so the shared training YAML and other launchers are untouched.

**Tech Stack:** Bash, Slurm, Accelerate, Python YAML override parsing.

## Global Constraints

- Modify only `submit_h200_i2v_lingbot_optim_requeue.sh`.
- Add exactly `training.learning_rate=1.0e-6`.
- Use exactly `logging.output_dir=outputs/i2v_lingbot_optim`.
- Do not rename the script, alter Slurm resources, change resume behavior, update the shared YAML, or add a new test case.

---

### Task 1: Update and validate the LingBot I2V launcher

**Files:**
- Modify: `submit_h200_i2v_lingbot_optim_requeue.sh:41-49`
- Modify: `tests/test_submit_i2v_lingbot_optim_requeue.py:17`

**Interfaces:**
- Consumes: the existing `train_i2v.py` override interface, where `training.learning_rate` and `logging.output_dir` are dot-path YAML overrides.
- Produces: a launcher that passes the requested learning rate and isolated output directory to `train_i2v.py`.

- [ ] **Step 1: Update the train command overrides**

  Add this override alongside the existing training overrides:

  ```bash
  training.learning_rate=1.0e-6 \\
  ```

  Replace the output override with:

  ```bash
  logging.output_dir=outputs/i2v_lingbot_optim \\
  ```

- [ ] **Step 2: Update the existing launcher assertion**

  In `test_i2v_lingbot_optim_requeue_script_uses_requested_overrides`, replace
  the old output-directory expectation and add the learning-rate expectation:

  ```python
  assert "training.learning_rate=1.0e-6" in script
  assert "logging.output_dir=outputs/i2v_lingbot_optim" in script
  ```

  Do not create a new test function.

- [ ] **Step 3: Inspect the exact launcher configuration**

  Run:

  ```bash
  rg -n "training.learning_rate|logging.output_dir" submit_h200_i2v_lingbot_optim_requeue.sh
  ```

  Expected output includes exactly `training.learning_rate=1.0e-6` and `logging.output_dir=outputs/i2v_lingbot_optim`, with no `i2v_lingbot_optim_requeue` output override.

- [ ] **Step 4: Validate the focused test and shell syntax**

  Run:

  ```bash
  PYTHONPATH=. conda run -n das pytest -q tests/test_submit_i2v_lingbot_optim_requeue.py
  bash -n submit_h200_i2v_lingbot_optim_requeue.sh
  ```

  Expected output: the test passes, then Bash produces no output and exits with status `0`.

- [ ] **Step 5: Commit the launcher change**

  ```bash
  git add submit_h200_i2v_lingbot_optim_requeue.sh tests/test_submit_i2v_lingbot_optim_requeue.py
  git commit -m "chore: tune LingBot I2V launcher overrides"
  ```
