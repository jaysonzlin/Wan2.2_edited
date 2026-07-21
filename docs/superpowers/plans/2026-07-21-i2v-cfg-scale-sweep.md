# I2V CFG-scale sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a deterministic TI2V sample for each requested CFG scale at one scheduler configuration.

**Architecture:** Keep the static sweep configuration in `sweep_i2v_inference.py`. The script will iterate over `EXPERIMENTS` and `CFG_SCALES`, applying each scale in the existing classifier-free-guidance call and naming the resulting MP4 with the scale. The GPU-free `--list-experiments` test defines the resulting run list.

**Tech Stack:** Python, pytest, existing Wan TI2V sampler.

## Global Constraints

- `EXPERIMENTS` must equal `((1, 50),)`.
- `CFG_SCALES` must equal `(0, 0.5, 0.75, 1, 2, 5)`.
- Each output filename must include its CFG scale and no result may be overwritten.
- The existing checkpoint loading and scheduler behavior must remain unchanged.

---

### Task 1: Add and verify the CFG-scale sweep

**Files:**
- Modify: `tests/test_sweep_i2v_inference.py:11-30`
- Modify: `sweep_i2v_inference.py:12-134`

**Interfaces:**
- Consumes: `EXPERIMENTS: tuple[tuple[int, int], ...]` and `CFG_SCALES: tuple[float, ...]`.
- Produces: `output_name(shift: int, num_steps: int, cfg_scale: float) -> str` and six dry-run output paths.

- [ ] **Step 1: Write the failing test**

```python
assert result.stdout.splitlines() == [
    "shift_1_steps_50_cfg_0.mp4",
    "shift_1_steps_50_cfg_0.5.mp4",
    "shift_1_steps_50_cfg_0.75.mp4",
    "shift_1_steps_50_cfg_1.mp4",
    "shift_1_steps_50_cfg_2.mp4",
    "shift_1_steps_50_cfg_5.mp4",
]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep_i2v_inference.py::test_lists_the_separate_shift_and_step_experiments -v`

Expected: FAIL because the script currently prints the old shift/step filenames.

- [ ] **Step 3: Write minimal implementation**

```python
CFG_SCALES = (0, 0.5, 0.75, 1, 2, 5)
EXPERIMENTS = ((1, 50),)

def output_name(shift: int, num_steps: int, cfg_scale: float) -> str:
    return f"shift_{shift}_steps_{num_steps}_cfg_{cfg_scale:g}.mp4"
```

Update both the dry-run loop and sampling loop to iterate over `CFG_SCALES`; reset the seeded generator, latent, and scheduler inside the scale loop; pass `cfg_scale` to `classifier_free_guidance`; and use the three-argument `output_name` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sweep_i2v_inference.py -v`

Expected: PASS with both sweep tests succeeding without CUDA.

- [ ] **Step 5: Commit**

```bash
git add sweep_i2v_inference.py tests/test_sweep_i2v_inference.py
git commit -m "feat: sweep I2V CFG scales"
```
