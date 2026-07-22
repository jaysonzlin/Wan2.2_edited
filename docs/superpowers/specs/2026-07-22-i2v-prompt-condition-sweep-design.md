# I2V prompt-condition sweep

## Purpose

Compare the fine-tuned Wan TI2V checkpoint under two fixed text-conditioning
experiments while holding the input image, seed, scheduler settings, checkpoint,
and CFG-scale sweep constant.

## Experiments

`sweep_i2v_inference.py` will define two named conditions and run each for every
entry in `EXPERIMENTS` and `CFG_SCALES`:

1. `no_prompt`: pass an empty string to both the conditional and CFG baseline
   text-encoder branches. This is image-to-video sampling without a supplied
   language prompt.
2. `standard_negative`: pass the existing Kubric positive prompt to the
   conditional branch and Wan's configured `sample_neg_prompt` string to the CFG
   baseline branch. The negative string is read from the TI2V configuration so
   it stays identical to the standard Wan image-to-video default.

The script will encode the two branch prompts separately and continue to compute
CFG as `baseline + scale * (conditional - baseline)`.

## Output contract

Every output filename will include the condition name before the existing
scheduler and CFG fields, preventing the two experiments from overwriting one
another. For example:

```text
no_prompt_shift_1_steps_50_cfg_1.mp4
standard_negative_shift_1_steps_50_cfg_1.mp4
```

`--list-experiments` remains GPU-free and prints all condition, scheduler, and
CFG-scale combinations in deterministic condition-first order.

## Scope and verification

The scheduler configuration, checkpoint loading, image conditioning, seed,
frame count, output codec, and existing CFG-scale values remain unchanged. The
test suite will verify the full planned output list and keep the scheduler's
first-sigma configuration test intact. No training behavior or CLI arguments
change.
