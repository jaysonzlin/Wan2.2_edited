# I2V LingBot Optimizer Requeue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an empty-prompt I2V H200 requeue job that uses LingBot's AdamW,
weight-decay, and gradient-clipping values without changing existing runs.

**Architecture:** Extract the small AdamW construction step in `train_i2v.py`
into a testable helper that consumes the `training` configuration. Define
explicit PyTorch-default AdamW fields in the base I2V YAML, then create a
separate launcher that overrides those fields and the existing clip/decay
settings with LingBot values.

**Tech Stack:** Python 3, PyTorch, OmegaConf configuration, Bash, pytest,
Conda environment `das`.

## Global Constraints

- The standard I2V config must retain AdamW `betas=(0.9, 0.999)`,
  `eps=1e-8`, weight decay `0.01`, and clip norm `1.0`.
- The new job retains `data.prompt=""` and invokes `train_i2v.py`.
- The new job uses `max_grad_norm=2.0`, `weight_decay=0.1`,
  `adam_beta1=0.9`, and `adam_beta2=0.95`.
- The new output directory is `outputs/i2v_lingbot_optim_requeue`.
- Slurm job and log identifiers are `wan_i2v_lingbot_optim`.
- Preserve H200, requeue, resume, checkpoint, and constant-LR settings from
  `submit_h200_i2v_requeue.sh`.
- Do not alter the user's untracked dataset/output files or deleted
  `logs/.gitkeep`.

## File structure

- Modify `train_i2v.py`: expose one focused AdamW factory and use it in main.
- Modify `configs/train/overfit_kubric_i2v.yaml`: make AdamW defaults explicit.
- Modify `tests/test_train_i2v.py`: test the factory's parameter group.
- Create `submit_h200_i2v_lingbot_optim_requeue.sh`: submit the isolated job.
- Create `tests/test_submit_i2v_lingbot_optim_requeue.py`: assert launcher
  behavior without Slurm.

---

### Task 1: Make I2V AdamW parameters configurable and testable

**Files:**
- Modify: `train_i2v.py:34-47, 211`
- Modify: `configs/train/overfit_kubric_i2v.yaml:20-23`
- Modify: `tests/test_train_i2v.py:7-25`

**Interfaces:**
- Produces: `create_optimizer(parameters, training) -> torch.optim.AdamW`.
- Consumes: mapping-like `training` with `learning_rate`, `weight_decay`,
  `adam_beta1`, `adam_beta2`, and `adam_epsilon` keys.
- Produces: an AdamW parameter group with
  `betas=(training["adam_beta1"], training["adam_beta2"])`,
  `eps=training["adam_epsilon"]`, and
  `weight_decay=training["weight_decay"]`.

- [x] **Step 1: Write the failing AdamW factory test**

  Add `import torch` and import `create_optimizer` in
  `tests/test_train_i2v.py`, then add:

  ```python
  def test_create_optimizer_uses_configured_adamw_parameters(self):
      parameter = torch.nn.Parameter(torch.zeros(()))
      optimizer = create_optimizer(
          [parameter],
          {
              "learning_rate": 1.0e-5,
              "weight_decay": 0.1,
              "adam_beta1": 0.9,
              "adam_beta2": 0.95,
              "adam_epsilon": 1.0e-8,
          },
      )

      group = optimizer.param_groups[0]
      self.assertEqual(group["betas"], (0.9, 0.95))
      self.assertEqual(group["eps"], 1.0e-8)
      self.assertEqual(group["weight_decay"], 0.1)
  ```

- [x] **Step 2: Run the test and confirm it fails**

  Run:

  ```bash
  conda run -n das python -m pytest \
      tests/test_train_i2v.py::TrainI2VHelperTests::test_create_optimizer_uses_configured_adamw_parameters -q
  ```

  Expected: import failure because `create_optimizer` has not been defined.

- [x] **Step 3: Add explicit AdamW defaults to the base configuration**

  Insert below `weight_decay: 0.01` in
  `configs/train/overfit_kubric_i2v.yaml`:

  ```yaml
  # Match PyTorch AdamW defaults unless a training run overrides them.
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  ```

- [x] **Step 4: Add the minimal optimizer factory and consume it**

  Add this helper below `resolve_unconditional_prompt` in `train_i2v.py`:

  ```python
  def create_optimizer(parameters, training) -> torch.optim.AdamW:
      """Create AdamW using the configured I2V optimizer parameters."""
      return torch.optim.AdamW(
          parameters,
          lr=training["learning_rate"],
          betas=(training["adam_beta1"], training["adam_beta2"]),
          eps=training["adam_epsilon"],
          weight_decay=training["weight_decay"],
      )
  ```

  Replace the inline `torch.optim.AdamW(...)` call in `main` with:

  ```python
  optimizer = create_optimizer(model.parameters(), training)
  ```

- [x] **Step 5: Run the focused training tests**

  Run:

  ```bash
  conda run -n das python -m pytest tests/test_train_i2v.py -q
  ```

  Expected: all tests in `tests/test_train_i2v.py` pass.

- [x] **Step 6: Commit the configurable optimizer support**

  ```bash
  git add train_i2v.py configs/train/overfit_kubric_i2v.yaml tests/test_train_i2v.py
  git commit -m "feat: configure I2V AdamW parameters"
  ```

### Task 2: Add the LingBot-aligned I2V launcher

**Files:**
- Create: `submit_h200_i2v_lingbot_optim_requeue.sh`
- Create: `tests/test_submit_i2v_lingbot_optim_requeue.py`

**Interfaces:**
- Consumes: Task 1 configuration fields accepted by `train_i2v.py`.
- Produces: an executable-style Bash submission file whose overrides select
  LingBot's beta/decay/clip settings.

- [x] **Step 1: Write the failing submission-script test**

  Create `tests/test_submit_i2v_lingbot_optim_requeue.py`:

  ```python
  from pathlib import Path


  def test_i2v_lingbot_optimizer_requeue_script_uses_requested_overrides() -> None:
      script = Path("submit_h200_i2v_lingbot_optim_requeue.sh").read_text()

      assert "#SBATCH --job-name=wan_i2v_lingbot_optim" in script
      assert "train_i2v.py" in script
      assert 'data.prompt=""' in script
      assert "training.max_grad_norm=2.0" in script
      assert "training.weight_decay=0.1" in script
      assert "training.adam_beta1=0.9" in script
      assert "training.adam_beta2=0.95" in script
      assert "logging.output_dir=outputs/i2v_lingbot_optim_requeue" in script
      assert "train_pc.py" not in script
  ```

- [x] **Step 2: Run the test and confirm it fails**

  Run:

  ```bash
  conda run -n das python -m pytest \
      tests/test_submit_i2v_lingbot_optim_requeue.py -q
  ```

  Expected: `FileNotFoundError` because the launcher does not yet exist.

- [x] **Step 3: Create the launcher**

  Copy the resource directives, container command, and existing overrides
  from `submit_h200_i2v_requeue.sh`. Name the result
  `submit_h200_i2v_lingbot_optim_requeue.sh`, replace every Slurm job/log
  identifier with `wan_i2v_lingbot_optim`, and use this command tail:

  ```bash
  train_i2v.py \
      --config configs/train/overfit_kubric_i2v.yaml \
      data.prompt="" \
      logging.output_dir=outputs/i2v_lingbot_optim_requeue \
      training.resume_from_checkpoint=latest \
      training.max_train_steps=10000 \
      training.checkpoint_every_steps=250 \
      training.checkpoints_total_limit=3 \
      training.lr_scheduler=constant \
      training.max_grad_norm=2.0 \
      training.weight_decay=0.1 \
      training.adam_beta1=0.9 \
      training.adam_beta2=0.95
  ```

- [x] **Step 4: Validate the new launcher**

  Run:

  ```bash
  bash -n submit_h200_i2v_lingbot_optim_requeue.sh
  conda run -n das python -m pytest \
      tests/test_submit_i2v_lingbot_optim_requeue.py -q
  ```

  Expected: Bash exits 0 and pytest reports one passing test.

- [x] **Step 5: Commit the isolated experiment launcher**

  ```bash
  git add submit_h200_i2v_lingbot_optim_requeue.sh \
      tests/test_submit_i2v_lingbot_optim_requeue.py
  git commit -m "feat: add LingBot optimizer I2V requeue job"
  ```

### Task 3: Verify the complete change

**Files:**
- Verify: all files listed in Tasks 1 and 2.

**Interfaces:**
- Consumes: the base config, optimizer factory, and isolated H200 launcher.
- Produces: repository-level regression evidence.

- [x] **Step 1: Run the full test suite**

  Run:

  ```bash
  MPLCONFIGDIR=/private/tmp/wan_matplotlib_cache conda run -n das python -m pytest -q
  ```

  Expected: all tests pass; existing Matplotlib/Pyparsing deprecation warnings
  may be emitted.

- [x] **Step 2: Inspect the final scope**

  Run:

  ```bash
  git diff --check
  git status --short
  git log --oneline --decorate -6
  ```

  Expected: no whitespace errors; only the intended feature commits and the
  pre-existing user changes named in Global Constraints remain.
