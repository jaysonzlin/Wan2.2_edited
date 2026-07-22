# Visualization Latent MSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log masked MSE between each periodic visualization's final sampled latent and its clean ground-truth latent.

**Architecture:** Add a pure masked-latent-MSE helper to the I2V training utilities. Make the existing visualization sampler return its final latent, then calculate and log the helper result only when the existing main-process visualization branch runs.

**Tech Stack:** Python 3, PyTorch, Accelerate, pytest/unittest, Wan TI2V.

## Global Constraints

- Do not change the flow-matching loss, optimizer, scheduler, seed, prompt conditioning, or 50-step visualization sampling process.
- Compare the visualization's final sampled latent directly with `clean_latents[0]`.
- Exclude latent time index `0`; include every future time index, channel, height, and width.
- Log only from the existing `accelerator.is_main_process` visualization branch as `train/visualization_denoised_latent_mse` at its current `global_step`.

---

### Task 1: Add a mask-aware latent MSE helper

**Files:**

- Modify: `training/wan_i2v_training.py:87-92`
- Test: `tests/test_wan_i2v_training.py:1-106`

**Interfaces:**

- Consumes: predicted and ground-truth latent tensors of shape `[B, C, T, H, W]`.
- Produces: `denoised_latent_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor`, a scalar that excludes `T=0`.

- [ ] **Step 1: Write the failing tests**

Import `denoised_latent_mse` and add:

```python
    def test_denoised_latent_mse_excludes_the_first_latent_slot(self):
        prediction = torch.tensor([[[[[100.0]], [[3.0]], [[5.0]]]]])
        target = torch.tensor([[[[[0.0]], [[1.0]], [[2.0]]]]])

        result = denoised_latent_mse(prediction, target)

        self.assertAlmostEqual(result.item(), 6.5)

    def test_denoised_latent_mse_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            denoised_latent_mse(torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 3, 1, 1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run `conda run -n das python -m pytest tests/test_wan_i2v_training.py -v`.

Expected: FAIL at collection because `denoised_latent_mse` is not exported.

- [ ] **Step 3: Write the minimal implementation**

Add this helper below `masked_velocity_mse`:

```python
def denoised_latent_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return latent MSE over denoised slots, excluding the clean input slot."""
    if prediction.shape != target.shape:
        raise ValueError("Predicted and target latent shapes must match")
    if prediction.ndim != 5 or prediction.shape[2] < 2:
        raise ValueError("Latents must have shape [B, C, T, H, W] with at least two time slots")
    return (prediction[:, :, 1:].float() - target[:, :, 1:].float()).square().mean()
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run `conda run -n das python -m pytest tests/test_wan_i2v_training.py -v`.

Expected: PASS, including the first-slot exclusion and shape-validation tests.

- [ ] **Step 5: Commit the helper**

Run `git add training/wan_i2v_training.py tests/test_wan_i2v_training.py` and `git commit -m "feat: add denoised latent MSE helper"`.

### Task 2: Return and log the visualization's final sampled latent

**Files:**

- Modify: `train_i2v.py:13-22, 75-151, 236-270`
- Test: `tests/test_train_i2v.py:1-73`

**Interfaces:**

- Consumes: `denoised_latent_mse`, `clean_latents[0]`, and the `latent` already produced by `save_visualization`.
- Produces: a `save_visualization(...) -> torch.Tensor` return value and one `accelerator.log` metric at each visualization event.

- [ ] **Step 1: Write the failing source-contract test**

Add this test to `tests/test_train_i2v.py`:

```python
    def test_visualization_metrics_are_logged_only_with_the_visualization(self):
        source = Path("train_i2v.py").read_text()

        self.assertIn("return latent", source)
        self.assertIn("denoised_latent_mse(visualization_latent, clean_latents[0])", source)
        self.assertIn('"train/visualization_denoised_latent_mse"', source)
```

- [ ] **Step 2: Run the test to verify it fails**

Run `conda run -n das python -m pytest tests/test_train_i2v.py::TrainI2VHelperTests::test_visualization_metrics_are_logged_only_with_the_visualization -v`.

Expected: FAIL because the visualization function returns `None` and no visualization latent metric exists.

- [ ] **Step 3: Write the minimal implementation**

Import `denoised_latent_mse` from `training.wan_i2v_training`. After the MP4 writer closes in `save_visualization`, restore train mode if necessary and return the final `latent` tensor:

```python
    if was_training:
        model.train()
    return latent
```

At the existing `if accelerator.is_main_process and global_step % training["visualization_every_steps"] == 0:` branch, store the return value and log the metric immediately after the visualization call:

```python
                visualization_latent = save_visualization(...)
                accelerator.log(
                    {
                        "train/visualization_denoised_latent_mse": denoised_latent_mse(
                            visualization_latent, clean_latents[0]
                        ).item()
                    },
                    step=global_step,
                )
```

Keep the existing visualization arguments and output path unchanged.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run `conda run -n das python -m pytest tests/test_train_i2v.py tests/test_wan_i2v_training.py -v`.

Expected: PASS, including the source contract and pure latent-MSE behavior.

- [ ] **Step 5: Commit the visualization integration**

Run `git add train_i2v.py tests/test_train_i2v.py` and `git commit -m "feat: log visualization denoised latent MSE"`.

## Final verification

- [ ] Run `conda run -n das python -m pytest -q` and confirm the full suite passes.
- [ ] Run `git diff --check` and confirm no whitespace errors.
