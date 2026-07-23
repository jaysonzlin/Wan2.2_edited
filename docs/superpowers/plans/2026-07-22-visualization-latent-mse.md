# Sampled Denoised-Latent MSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log a full-sampling denoised-latent MSE every 10 optimizer steps while retaining MP4 visualization output every 250 steps.

**Architecture:** Keep `denoised_latent_mse` as the pure masked comparison. Split sampling from MP4 writing so the trainer can sample for metrics without producing a video, and reuse the same sampled latent when a metric and visualization cadence coincide.

**Tech Stack:** Python 3, PyTorch, Accelerate, unittest/pytest, Wan TI2V.

## Global Constraints

- Add `training.denoised_latent_mse_every_steps: 10`; it must be a positive integer.
- Preserve the 50-step solver, seed, prompts, CFG scale, time shift, MP4 path, and `visualization_every_steps` behavior.
- Log only `train/denoised_latent_mse`; remove `train/visualization_denoised_latent_mse`.
- Compare the sampled latent with the current local batch's first clean latent, excluding latent time index `0`.
- At a step matching both cadences, sample once and use that latent both for logging and MP4 writing.

---

### Task 1: Configure and validate the metric cadence

**Files:**
- Modify: `configs/train/overfit_kubric_i2v.yaml:40-44`
- Modify: `training/overfit_config.py:49-59`
- Modify: `tests/test_overfit_config.py:42-57`

**Interfaces:**
- Consumes: `config["training"]`.
- Produces: a required positive integer `training["denoised_latent_mse_every_steps"]` available to `train_i2v.main`.

- [ ] **Step 1: Write the failing validation test**

  Add to `OverfitConfigTests`:

  ```python
      def test_invalid_denoised_latent_mse_cadence_is_rejected(self):
          with tempfile.TemporaryDirectory() as temporary_directory:
              path = write_yaml(
                  Path(temporary_directory),
                  {"training": {"denoised_latent_mse_every_steps": 0}},
              )

              with self.assertRaisesRegex(
                  ValueError, "denoised_latent_mse_every_steps must be a positive integer"
              ):
                  load_config(path, [])
  ```

- [ ] **Step 2: Verify the test fails for the missing validation**

  Run: `python -m pytest tests/test_overfit_config.py::OverfitConfigTests::test_invalid_denoised_latent_mse_cadence_is_rejected -v`

  Expected: FAIL because `load_config` accepts zero.

- [ ] **Step 3: Add the default and validation**

  Under `visualization_every_steps` in `configs/train/overfit_kubric_i2v.yaml`, add:

  ```yaml
  denoised_latent_mse_every_steps: 10
  ```

  Expand the `validate_config` cadence-key tuple in `training/overfit_config.py` to:

  ```python
      for key in (
          "max_train_steps",
          "warmup_steps",
          "checkpoint_every_steps",
          "denoised_latent_mse_every_steps",
      ):
  ```

  This existing validation produces the required error for `0` and non-integers.

- [ ] **Step 4: Verify the focused configuration tests pass**

  Run: `python -m pytest tests/test_overfit_config.py -v`

  Expected: PASS, including the new zero-cadence rejection.

- [ ] **Step 5: Commit the configuration task**

  ```bash
  git add configs/train/overfit_kubric_i2v.yaml training/overfit_config.py tests/test_overfit_config.py
  git commit -m "feat: configure sampled latent MSE cadence"
  ```

### Task 2: Separate latent sampling from MP4 output

**Files:**
- Modify: `train_i2v.py:81-150`
- Modify: `tests/test_train_i2v.py:102-116`

**Interfaces:**
- Produces: `sample_visualization_latent(model, vae, text_encoder, condition_frame, prompt, unconditional_prompt, wan_config, time_shift, num_frames, seed, cfg_scale) -> torch.Tensor`.
- Produces: `save_visualization(vae, latent, output_file, fps) -> None`.
- `sample_visualization_latent` owns model mode restoration; `save_visualization` only decodes and writes the supplied latent.

- [ ] **Step 1: Write the failing sampler-separation source-contract test**

  Replace `test_visualization_metrics_are_logged_only_with_the_visualization` with:

  ```python
      def test_sampling_is_separate_from_visualization_output(self):
          source = Path("train_i2v.py").read_text()

          self.assertIn("def sample_visualization_latent(", source)
          self.assertIn("def save_visualization(", source)
          self.assertIn("return latent", source)
  ```

- [ ] **Step 2: Verify it fails for the current combined function**

  Run: `python -m pytest tests/test_train_i2v.py::TrainI2VHelperTests::test_sampling_is_separate_from_visualization_output -v`

  Expected: FAIL because `sample_visualization_latent` does not exist and `save_visualization` has the old signature.

- [ ] **Step 3: Split the sampler and writer**

  Rename the current decorated `save_visualization` function to
  `sample_visualization_latent`, remove its `output_file` parameter, and retain
  its model-evaluation, solver loop, `if was_training: model.train()`, and
  `return latent` behavior.  Its signature must be:

  ```python
  @torch.no_grad()
  def sample_visualization_latent(
      model, vae, text_encoder, condition_frame, prompt, unconditional_prompt,
      wan_config, time_shift, num_frames, seed, cfg_scale,
  ) -> torch.Tensor:
  ```

  Immediately below it, add the writer that contains only the old decode and
  MP4-output tail:

  ```python
  def save_visualization(
      vae, latent: torch.Tensor, output_file: Path, fps: int
  ) -> None:
      """Decode a sampled latent and write the local qualitative MP4."""
      from imageio.v2 import get_writer

      video = vae.decode([latent])[0].permute(1, 2, 3, 0)
      frames = ((video.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
      output_file.parent.mkdir(parents=True, exist_ok=True)
      with get_writer(output_file, fps=fps, codec="libx264", quality=8) as writer:
          for frame in frames:
              writer.append_data(frame)
  ```

  Its call must pass `ti2v_5B.sample_fps`.

- [ ] **Step 4: Verify the focused sampler contract passes**

  Run: `python -m pytest tests/test_train_i2v.py::TrainI2VHelperTests::test_sampling_is_separate_from_visualization_output -v`

  Expected: PASS.

- [ ] **Step 5: Commit the sampler split**

  ```bash
  git add train_i2v.py tests/test_train_i2v.py
  git commit -m "refactor: separate I2V sampling from video output"
  ```

### Task 3: Log sampled MSE independently of video cadence

**Files:**
- Modify: `train_i2v.py:279-306`
- Modify: `tests/test_train_i2v.py:102-125`

**Interfaces:**
- Consumes: `training["denoised_latent_mse_every_steps"]`, `sample_visualization_latent`, `save_visualization`, `denoised_latent_mse`, and current-batch `clean_latents[0]`.
- Produces: `train/denoised_latent_mse` at every matching optimizer step and an MP4 only at matching visualization steps.

- [ ] **Step 1: Write the failing cadence source-contract test**

  Add to `TrainI2VHelperTests`:

  ```python
      def test_sampled_latent_mse_has_its_own_cadence_and_reuses_visualization_sample(self):
          compact_source = "".join(Path("train_i2v.py").read_text().split())

          self.assertIn('global_step%training["denoised_latent_mse_every_steps"]==0', compact_source)
          self.assertIn('"train/denoised_latent_mse"', compact_source)
          self.assertNotIn('"train/visualization_denoised_latent_mse"', compact_source)
          self.assertIn('sampled_latent=sample_visualization_latent(', compact_source)
          self.assertIn('save_visualization(vae,sampled_latent,', compact_source)
  ```

- [ ] **Step 2: Verify it fails for the visualization-only implementation**

  Run: `python -m pytest tests/test_train_i2v.py::TrainI2VHelperTests::test_sampled_latent_mse_has_its_own_cadence_and_reuses_visualization_sample -v`

  Expected: FAIL because the trainer only samples and logs inside the 250-step visualization branch.

- [ ] **Step 3: Implement independent sampling, logging, and MP4 writing**

  Replace the existing visualization block with:

  ```python
            should_log_denoised_mse = (
                global_step % training["denoised_latent_mse_every_steps"] == 0
            )
            should_save_visualization = (
                accelerator.is_main_process
                and global_step % training["visualization_every_steps"] == 0
            )
            if should_log_denoised_mse or should_save_visualization:
                sampled_latent = sample_visualization_latent(
                    accelerator.unwrap_model(model), vae, text_encoder, videos[0, 0],
                    data["prompt"], unconditional_prompt, ti2v_5B,
                    training["time_shift"], data["num_frames"], training["seed"],
                    training["visualization_cfg_scale"],
                )
                if should_log_denoised_mse:
                    accelerator.log(
                        {
                            "train/denoised_latent_mse": denoised_latent_mse(
                                sampled_latent, clean_latents[0]
                            ).item()
                        },
                        step=global_step,
                    )
                if should_save_visualization:
                    save_visualization(
                        vae, sampled_latent, visualization_path(output_dir, epoch),
                        ti2v_5B.sample_fps,
                    )
  ```

  Keep this block after checkpoint handling.  Do not change the regular loss,
  learning-rate, or gradient-norm logging.

- [ ] **Step 4: Verify focused trainer and helper tests pass**

  Run: `python -m pytest tests/test_train_i2v.py tests/test_wan_i2v_training.py -v`

  Expected: PASS, including first-slot masking and the new independent cadence
  source contract.

- [ ] **Step 5: Commit the metric integration**

  ```bash
  git add train_i2v.py tests/test_train_i2v.py
  git commit -m "feat: log sampled denoised latent MSE"
  ```

### Task 4: Align design records and run the full focused suite

**Files:**
- Modify: `docs/superpowers/specs/2026-07-22-visualization-latent-mse-design.md`
- Modify: `docs/superpowers/plans/2026-07-22-visualization-latent-mse.md`

**Interfaces:**
- Consumes: the completed configuration, sampler split, and metric integration.
- Produces: documentation that states the implemented 10-step sampling metric and 250-step MP4 behavior.

- [ ] **Step 1: Check the records name the final behavior**

  Confirm both documents state all of: a positive configurable
  `denoised_latent_mse_every_steps` default of `10`, full sampling without MP4
  at metric events, and reuse of the sampled latent for an MP4 at visualization
  events.

- [ ] **Step 2: Run the full focused regression suite**

  Run: `python -m pytest tests/test_overfit_config.py tests/test_train_i2v.py tests/test_wan_i2v_training.py -v`

  Expected: PASS with no collection errors or test failures.

- [ ] **Step 3: Inspect the final diff**

  Run: `git diff --check HEAD~3..HEAD && git status --short`

  Expected: no whitespace errors; only intended tracked changes or documented pre-existing untracked files.

- [ ] **Step 4: Commit the aligned documentation**

  ```bash
  git add docs/superpowers/specs/2026-07-22-visualization-latent-mse-design.md docs/superpowers/plans/2026-07-22-visualization-latent-mse.md
  git commit -m "docs: plan sampled latent MSE logging"
  ```
