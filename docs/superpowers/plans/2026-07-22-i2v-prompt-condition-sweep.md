# I2V Prompt-Condition Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample the fine-tuned TI2V checkpoint under a no-prompt condition and a standard-Wan-negative-prompt CFG condition for every existing scheduler and CFG-scale experiment.

**Architecture:** Keep `sweep_i2v_inference.py` as the single inference entrypoint. Add an immutable condition table that supplies the conditional and CFG-baseline prompt for each named experiment. Derive filenames and the GPU-free listing from that table, then encode the selected pair in the existing sampling loop.

**Tech Stack:** Python 3, pytest, PyTorch, Wan TI2V configuration, imageio.

## Global Constraints

- Keep `EXPERIMENTS`, `CFG_SCALES`, checkpoint loading, seed, image conditioning, scheduler setup, frame count, and video encoding unchanged.
- `no_prompt` encodes `""` for both the conditional and CFG-baseline branches.
- `standard_negative` encodes `PROMPT` conditionally and `ti2v_5B.sample_neg_prompt` as its CFG-baseline branch.
- `--list-experiments` stays GPU-free and lists condition-first, scheduler-second, CFG-scale-third.
- Output files include the condition name and cannot overwrite one another.

---

### Task 1: Define the prompt-condition sweep contract

**Files:**

- Modify: `sweep_i2v_inference.py:10-37`
- Test: `tests/test_sweep_i2v_inference.py:11-28`

**Interfaces:**

- Consumes: existing `PROMPT`, `EXPERIMENTS`, and `CFG_SCALES` constants.
- Produces: `PROMPT_CONDITIONS`, `output_name(condition, shift, num_steps, cfg_scale)`, and a 12-name `--list-experiments` output.

- [ ] **Step 1: Write the failing test**

Replace the listing assertion with:

```python
    assert result.stdout.splitlines() == [
        "no_prompt_shift_1_steps_50_cfg_0.mp4",
        "no_prompt_shift_1_steps_50_cfg_0.5.mp4",
        "no_prompt_shift_1_steps_50_cfg_0.75.mp4",
        "no_prompt_shift_1_steps_50_cfg_1.mp4",
        "no_prompt_shift_1_steps_50_cfg_2.mp4",
        "no_prompt_shift_1_steps_50_cfg_5.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.5.mp4",
        "standard_negative_shift_1_steps_50_cfg_0.75.mp4",
        "standard_negative_shift_1_steps_50_cfg_1.mp4",
        "standard_negative_shift_1_steps_50_cfg_2.mp4",
        "standard_negative_shift_1_steps_50_cfg_5.mp4",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run `pytest tests/test_sweep_i2v_inference.py::test_lists_the_cfg_scale_sweep_for_the_fixed_scheduler_experiment -v`.

Expected: FAIL because the current script lists only six filenames without a prompt-condition prefix.

- [ ] **Step 3: Write the minimal implementation**

Add this condition table and replace the existing output helper:

```python
PROMPT_CONDITIONS = (
    ("no_prompt", "", ""),
    ("standard_negative", PROMPT, None),
)


def output_name(condition: str, shift: int, num_steps: int, cfg_scale: float) -> str:
    """Return the per-sample MP4 filename for one prompt condition."""
    return f"{condition}_shift_{shift}_steps_{num_steps}_cfg_{cfg_scale:g}.mp4"
```

Change `--list-experiments` to:

```python
        for condition, _, _ in PROMPT_CONDITIONS:
            for shift, num_steps in EXPERIMENTS:
                for cfg_scale in CFG_SCALES:
                    print(output_name(condition, shift, num_steps, cfg_scale))
```

`None` marks the condition whose baseline prompt is read from `ti2v_5B.sample_neg_prompt` only after the Wan configuration import.

- [ ] **Step 4: Run the focused test to verify it passes**

Run `pytest tests/test_sweep_i2v_inference.py::test_lists_the_cfg_scale_sweep_for_the_fixed_scheduler_experiment -v`.

Expected: PASS, with all 12 filenames in the asserted order.

- [ ] **Step 5: Commit the contract change**

Run `git add sweep_i2v_inference.py tests/test_sweep_i2v_inference.py` and `git commit -m "feat: add I2V prompt condition sweep"`.

### Task 2: Apply each condition to the CFG sampling branches

**Files:**

- Modify: `sweep_i2v_inference.py:62-138`
- Test: `tests/test_sweep_i2v_inference.py:11-35`

**Interfaces:**

- Consumes: `PROMPT_CONDITIONS` and `ti2v_5B.sample_neg_prompt`.
- Produces: `resolve_prompt_pair(condition, standard_negative_prompt)` and a sampling loop that uses the required conditional and baseline text contexts and writes condition-prefixed names.

- [ ] **Step 1: Write the failing test**

Add this failing pure-function test:

```python
def test_resolves_no_prompt_and_standard_negative_contexts() -> None:
    assert sweep_i2v_inference.resolve_prompt_pair(
        ("no_prompt", "", ""), standard_negative_prompt="unused"
    ) == ("", "")
    assert sweep_i2v_inference.resolve_prompt_pair(
        ("standard_negative", sweep_i2v_inference.PROMPT, None),
        standard_negative_prompt="wan default negative",
    ) == (sweep_i2v_inference.PROMPT, "wan default negative")
```

- [ ] **Step 2: Run the test to verify it fails**

Run `pytest tests/test_sweep_i2v_inference.py::test_resolves_no_prompt_and_standard_negative_contexts -v`.

Expected: FAIL with `AttributeError` because `resolve_prompt_pair` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Add this helper beside `output_name`:

```python
def resolve_prompt_pair(
    condition: tuple[str, str, str | None], standard_negative_prompt: str
) -> tuple[str, str]:
    """Return the conditional and CFG-baseline prompts for one condition."""
    _, conditional_prompt, baseline_prompt = condition
    return conditional_prompt, standard_negative_prompt if baseline_prompt is None else baseline_prompt
```

Remove the single pre-loop context encoding. Wrap the existing scheduler/CFG loop in this condition loop:

```python
        for condition in PROMPT_CONDITIONS:
            condition_name, _, _ = condition
            conditional_prompt, baseline_prompt = resolve_prompt_pair(
                condition, ti2v_5B.sample_neg_prompt
            )
            conditional_context = text_encoder([conditional_prompt], device)
            baseline_context = text_encoder([baseline_prompt], device)

            for shift, num_steps, cfg_scale in (
                (shift, num_steps, cfg_scale)
                for shift, num_steps in EXPERIMENTS
                for cfg_scale in CFG_SCALES
            ):
```

Indent the latent initialization, scheduler setup, denoising, and video writing one level. Rename `unconditional` to `baseline`, call the model with `context=baseline_context`, pass `baseline` as the first argument of `classifier_free_guidance`, and set the output path with:

```python
            output_path = OUT_DIR / output_name(condition_name, shift, num_steps, cfg_scale)
```

Do not alter CFG math or any scheduler/model parameter.

- [ ] **Step 4: Run focused and full tests to verify they pass**

Run `pytest tests/test_sweep_i2v_inference.py -v`.

Expected: PASS for the output list, the prompt-pair resolver, and the existing first-sigma scheduler test.

- [ ] **Step 5: Run GPU-free command-line verification**

Run `python sweep_i2v_inference.py --list-experiments`.

Expected: the same 12 lines as Task 1, without CUDA or checkpoint access.

- [ ] **Step 6: Commit the sampling integration**

Run `git add sweep_i2v_inference.py tests/test_sweep_i2v_inference.py` and `git commit -m "feat: use standard negative prompt in I2V sweep"`.

## Final verification

- [ ] Run `pytest tests/test_sweep_i2v_inference.py -v` and confirm all tests pass.
- [ ] Run `python sweep_i2v_inference.py --list-experiments` and compare its 12 lines with Task 1.
- [ ] Run `git diff --check` and confirm there are no whitespace errors.
