# TI2V Negative-Prompt Requeue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TI2V H200 requeue training job that uses a fixed positive
prompt and Wan's canonical negative prompt for every unconditional branch.

**Architecture:** `train_i2v.py` will resolve one unconditional prompt from a
backward-compatible training option after loading the Wan configuration. The
same resolved text is encoded for dropped training contexts and for the
visualization CFG baseline. A dedicated submission script enables the option
and isolates all state from the existing empty-prompt I2V run.

**Tech Stack:** Python 3, PyTorch, Accelerate, OmegaConf configuration, Bash,
pytest, Conda environment `das`.

## Global Constraints

- Default behavior remains the empty unconditional prompt.
- The positive prompt is exactly `Colorful object falls onto the light gray ground`.
- The enabled unconditional prompt is `ti2v_5B.sample_neg_prompt`.
- The new output directory is `outputs/ti2v_negative_requeue`.
- The job and Slurm log prefix is `wan_ti2v`.
- Do not change the user's existing untracked files or deleted `logs/.gitkeep`.

## File structure

- Modify `train_i2v.py`: resolve and consume the unconditional prompt.
- Modify `configs/train/overfit_kubric_i2v.yaml`: define the default option.
- Create `submit_h200_ti2v_negative_requeue.sh`: launch the distinct H200 job.
- Modify `tests/test_train_i2v.py`: cover prompt resolution and code wiring.
- Create `tests/test_submit_ti2v_negative_requeue.py`: validate submission
  settings without a Slurm cluster.

---

### Task 1: Resolve and propagate the unconditional prompt

**Files:**
- Modify: `train_i2v.py:24-96, 150-236`
- Modify: `tests/test_train_i2v.py:7-73`

**Interfaces:**
- Produces: `resolve_unconditional_prompt(use_wan_negative_prompt: bool,
  wan_negative_prompt: str) -> str`.
- Consumes: `ti2v_5B.sample_neg_prompt` and
  `training.get("use_wan_negative_prompt", False)`.
- Produces: `save_visualization(..., prompt, unconditional_prompt, output_file,
  ...) -> torch.Tensor` with the new string parameter.

- [x] **Step 1: Write the failing resolver tests**

  Add `resolve_unconditional_prompt` to the `from train_i2v import (...)` list
  in `tests/test_train_i2v.py`, then add:

  ```python
  def test_unconditional_prompt_is_empty_by_default(self):
      self.assertEqual(resolve_unconditional_prompt(False, "Wan negative"), "")

  def test_unconditional_prompt_uses_wan_negative_prompt_when_enabled(self):
      self.assertEqual(
          resolve_unconditional_prompt(True, "Wan negative"), "Wan negative"
      )
  ```

- [x] **Step 2: Run the focused tests and confirm they fail**

  Run:

  ```bash
  conda run -n das python -m pytest tests/test_train_i2v.py -q
  ```

  Expected: import failure for `resolve_unconditional_prompt`.

- [x] **Step 3: Add the minimal resolver**

  Add this module-level helper below `visualization_path` in `train_i2v.py`:

  ```python
  def resolve_unconditional_prompt(
      use_wan_negative_prompt: bool, wan_negative_prompt: str
  ) -> str:
      """Return the text used by classifier-free unconditional branches."""
      return wan_negative_prompt if use_wan_negative_prompt else ""
  ```

- [x] **Step 4: Write the source-wiring regression test**

  Add a test to `tests/test_train_i2v.py` that compacts whitespace from
  `train_i2v.py` and asserts all of the following exact snippets:

  ```python
  "resolve_unconditional_prompt(training.get(\"use_wan_negative_prompt\",False),ti2v_5B.sample_neg_prompt)"
  "text_encoder([unconditional_prompt]*len(context),accelerator.device)"
  "data[\"prompt\"],unconditional_prompt,visualization_path"
  ```

- [x] **Step 5: Propagate the resolved string**

  In `main`, after `training, data, logging = ...`, set:

  ```python
  unconditional_prompt = resolve_unconditional_prompt(
      training.get("use_wan_negative_prompt", False), ti2v_5B.sample_neg_prompt
  )
  ```

  Change the dropout branch to encode the resolved text:

  ```python
  unconditional_context = text_encoder(
      [unconditional_prompt] * len(context), accelerator.device
  )
  context = apply_classifier_free_dropout(context, unconditional_context, drop_mask)
  ```

  Insert `unconditional_prompt` immediately after `prompt` in the
  `save_visualization` signature and call. Use it in:

  ```python
  unconditional_context = text_encoder([unconditional_prompt], device)
  ```

- [x] **Step 6: Run the focused training tests**

  Run:

  ```bash
  conda run -n das python -m pytest tests/test_train_i2v.py -q
  ```

  Expected: all tests pass.

- [x] **Step 7: Commit the independently tested prompt behavior**

  ```bash
  git add train_i2v.py tests/test_train_i2v.py
  git commit -m "feat: support Wan negative unconditional prompt"
  ```

### Task 2: Expose the option and add the TI2V submission job

**Files:**
- Modify: `configs/train/overfit_kubric_i2v.yaml:22-25`
- Create: `submit_h200_ti2v_negative_requeue.sh`
- Create: `tests/test_submit_ti2v_negative_requeue.py`

**Interfaces:**
- Consumes: `training.use_wan_negative_prompt` defined by Task 1.
- Produces: a Bash script that passes the fixed prompt, enabled option, and
  distinct output directory to `train_i2v.py`.

- [x] **Step 1: Write the failing submission-script test**

  Create `tests/test_submit_ti2v_negative_requeue.py`:

  ```python
  from pathlib import Path


  def test_ti2v_negative_requeue_script_uses_requested_prompt_and_output() -> None:
      script = Path("submit_h200_ti2v_negative_requeue.sh").read_text()

      assert "#SBATCH --job-name=wan_ti2v" in script
      assert "train_i2v.py" in script
      assert 'data.prompt="Colorful object falls onto the light gray ground"' in script
      assert "training.use_wan_negative_prompt=true" in script
      assert "logging.output_dir=outputs/ti2v_negative_requeue" in script
      assert "train_pc.py" not in script
  ```

- [x] **Step 2: Run the new test and confirm it fails**

  Run:

  ```bash
  conda run -n das python -m pytest tests/test_submit_ti2v_negative_requeue.py -q
  ```

  Expected: `FileNotFoundError` because the script has not been created.

- [x] **Step 3: Add the configuration default**

  Insert below `text_dropout_probability: 0.1` in
  `configs/train/overfit_kubric_i2v.yaml`:

  ```yaml
  # Use Wan's canonical negative prompt instead of empty text for unconditional branches.
  use_wan_negative_prompt: false
  ```

- [x] **Step 4: Create the submission script**

  Copy the resource settings, container invocation, and training overrides
  from `submit_h200_i2v_requeue.sh`. In
  `submit_h200_ti2v_negative_requeue.sh`, change all job and log identifiers
  from `wan_i2v` to `wan_ti2v`, and use this override tail:

  ```bash
  train_i2v.py \
      --config configs/train/overfit_kubric_i2v.yaml \
      data.prompt="Colorful object falls onto the light gray ground" \
      training.use_wan_negative_prompt=true \
      logging.output_dir=outputs/ti2v_negative_requeue \
      training.resume_from_checkpoint=latest \
      training.max_train_steps=10000 \
      training.checkpoint_every_steps=250 \
      training.checkpoints_total_limit=3 \
      training.lr_scheduler=constant
  ```

- [x] **Step 5: Run script and focused-test validation**

  Run:

  ```bash
  bash -n submit_h200_ti2v_negative_requeue.sh
  conda run -n das python -m pytest tests/test_submit_ti2v_negative_requeue.py -q
  ```

  Expected: Bash exits with code 0; pytest reports one passing test.

- [x] **Step 6: Commit the job and its configuration**

  ```bash
  git add configs/train/overfit_kubric_i2v.yaml \
      submit_h200_ti2v_negative_requeue.sh \
      tests/test_submit_ti2v_negative_requeue.py
  git commit -m "feat: add TI2V negative-prompt requeue job"
  ```

### Task 3: Verify the complete repository state

**Files:**
- Verify: all files from Tasks 1 and 2.

**Interfaces:**
- Consumes: completed prompt behavior and submission job.
- Produces: evidence that the integration preserves the full test suite.

- [x] **Step 1: Run the complete test suite**

  Run:

  ```bash
  MPLCONFIGDIR=/private/tmp/wan_matplotlib_cache conda run -n das python -m pytest -q
  ```

  Expected: all tests pass; existing non-failing Matplotlib deprecation
  warnings may appear.

- [x] **Step 2: Inspect the scoped change set**

  Run:

  ```bash
  git diff --check
  git status --short
  git log --oneline --decorate -5
  ```

  Expected: no whitespace errors, only the intended feature commits plus the
  pre-existing user changes listed in Global Constraints.
