# Joint Wan–PhysCtrl Training: Implementation Plan

> **For the implementer:** Follow the test-first order below. Keep the pre-existing dirty
> worktree changes untouched; this feature lives only in the named new/modified files.

**Goal:** Train Wan 2.2 TI2V 5B and a randomly initialized per-object PhysCtrl trajectory
branch jointly. Couple Wan's final eight transformer blocks with PhysCtrl's eight blocks
through full-token, bidirectional cross-attention, and provide synchronized joint sampling
plus the existing point-cloud comparison videos.

**Architecture:** A wrapper owns the existing Wan and PhysCtrl backbones. It executes their
native block phases to exchange state after Wan text attention and after PhysCtrl temporal
attention. At each paired layer, Wan tokens attend to the concatenation of all object point
tokens, while each object's tokens attend only the video's tokens. Projection into a shared
512-dimensional, eight-head attention space handles the unequal 3072/256 hidden widths.

**Tech stack:** PyTorch, Wan's Flow-UniPC I2V utilities, the existing PhysCtrl DDPM scheduler,
Hydra/OmegaConf YAML configuration, pytest.

## Non-negotiable training contract

- Load the base Wan 2.2 TI2V 5B checkpoint. Freeze only its VAE and text encoder; optimize all
  Wan DiT, PhysCtrl, and bridge parameters with one AdamW group using `lr=1e-5`,
  `betas=(0.9,0.95)`, `eps=1e-8`, and `weight_decay=0.1`.
- Use an empty prompt exclusively (`text_dropout_probability=0`).
- Train with `batch_size=1`. A clip may contain any number of lexical-sorted object directories;
  the initial curriculum config limits samples to `K <= 4`, but model/data code has no hard cap.
- Wan uses its current shifted flow-matching velocity loss. Each object uses only the PC DDPM
  x0 objective. Sample Wan's flow time and the PC timestep independently, but give every object
  in a clip the same PC timestep and independent Gaussian PC noise.
- Sum—not average—the valid per-object PC losses: `loss = video_loss + sum(object_pc_losses)`.
- Keep the frame-0 camera coordinate system and point-index correspondence fixed across all
  49 frames. Do not resample or reorder points.

## Sample layout

```
sample_0001/
  rgba_00000.png ... rgba_00048.png
  metadata.json
  objects/
    000/pc.hdf5
    001/pc.hdf5
```

`metadata.json` must be a parseable JSON object in v1. Each `pc.hdf5` supplies
`point_cloud`, `initial_linear_velocity`, and `initial_angular_velocity`, using the established
`(49,1,2048,3)` point-cloud representation.

## Implementation sequence

### 1. Define the joint configuration and multi-object dataset

**Files:**
- Create `training/joint_config.py`
- Create `training/joint_dataset.py`
- Create `configs/train/joint_wan_physctrl_832x480.yaml`
- Create `tests/test_joint_dataset.py`

1. Write failing tests that create two tiny on-disk samples with variable object counts, shuffled
   directory creation order, 49 image names, parseable `metadata.json`, and HDF5 trajectory keys.
   Assert lexical object order, trajectory/velocity shapes, preservation of point values, and
   rejection of malformed metadata, missing frames, and an inconsistent 49-frame trajectory.
2. Implement a dataset returning one video clip plus a variable-length object collection, with no
   padding requirement at batch size one. Reuse image transform conventions from the I2V dataset
   and HDF5 conventions from `training/pc_dataset.py`.
3. Implement config validation for the fixed dimensions, `batch_size=1`, empty prompt, 8 paired
   layers, 512 bridge width, 8 heads, PC DDPM x0 mode, and initial `max_objects_per_sample=4`
   curriculum guard. Do not add a `max_train_samples` option.
4. Run `pytest tests/test_joint_dataset.py`.

### 2. Add aligned multi-object PC DDPM batching and losses

**Files:**
- Create `training/joint_objectives.py`
- Create `tests/test_joint_objectives.py`

1. Write failing tests with a fake scheduler to prove that all objects get the same sampled
   integer timestep, different per-object noise, a target equal to clean x0, and frame times of
   the expected format. Add a numeric loss test establishing the exact sum of per-object MSEs,
   not their mean.
2. Implement a small batch dataclass and an aligned-PC-DDPM helper. It should strip the frame-0
   condition from the noised target inputs while retaining clean frame-0 points for conditioning;
   trajectory predictions cover frames 1--48.
3. Implement `per_object_pc_x0_mse`, returning one scalar per object and a separately named sum.
4. Run `pytest tests/test_joint_objectives.py`.

### 3. Expose safe block-phase seams in the existing backbones

**Files:**
- Modify `wan/modules/model.py`
- Modify `wan/modules/pc_trajectory.py`
- Modify `wan/modules/pc_physctrl.py` only if a small public phase helper is necessary
- Extend `tests/test_pc_trajectory_model.py`
- Create or extend a focused Wan block test under `tests/`

1. Write regression tests that compare the existing normal `WanAttentionBlock.forward` result
   against the result of its new self-attention, text-cross-attention, and MLP phase calls with
   no bridge. Add an analogous PC test proving that normal `PCTrajectoryModel.forward` remains
   numerically equivalent when its encode/block/decode helpers are used.
2. Refactor—not change—the native sequence:
   - Wan: self attention, then text cross attention, then MLP.
   - PhysCtrl: spatial attention, MLP, then temporal attention.
3. Make the wrapper-visible state carry all existing modulation/time inputs so it can perform an
   exchange immediately after the two required native phases. Preserve the legacy model forward
   APIs for standalone training/inference.
4. Run the focused tests plus existing `tests/test_pc_trajectory_model.py`.

### 4. Build the bidirectional bridge and joint model wrapper

**Files:**
- Create `wan/modules/joint_wan_physctrl.py`
- Create `tests/test_joint_bridge.py`
- Update the appropriate `wan/modules/__init__.py` export file if needed

1. Write failing bridge tests for tensor shapes, independent zero-initialized residual gates,
   finite gradients through both directions, and no changes to either branch when gates are zero.
   Add a test with `K=1` and `K=3` that proves the video sees concatenated object keys/values while
   each object receives only the video sequence.
2. Implement `BidirectionalWanPhysCtrlBridge` with independent Q/K/V/output projections
   (`3072 <-> 512`, `256 <-> 512`) and 8-head scaled-dot-product attention. Use a CUDA-capable
   PyTorch attention path; do not pool or cap tokens.
3. Implement `JointWanPhysCtrlModel`: run the first 22 Wan blocks normally, pair Wan blocks
   22--29 with PhysCtrl blocks 0--7, and decode video velocity and one x0 trajectory per object.
   Group object work internally where useful, but retain object membership for the one-to-video
   attention rule. Enforce the initial batch-size-one path clearly.
4. Run `pytest tests/test_joint_bridge.py` and the regression tests from step 3.

### 5. Implement synchronized joint inference and visualization

**Files:**
- Create `wan/joint_pc_pipeline.py`
- Create `tests/test_joint_pc_pipeline.py`

1. Write a failing fake-model/fake-scheduler test that records calls and verifies exactly one
   joint model evaluation per outer step, equal default 50-step Flow-UniPC/DDIM schedules, paired
   current states, PC x0 consumption by DDIM, and preservation of the I2V latent slice plus each
   initial point cloud condition.
2. Implement a joint pipeline that takes a condition image/video setup, empty text context,
   K initial point clouds, and their linear/angular velocities. At every outer step it calls the
   joint model once and updates Wan with Flow-UniPC and every object with DDIM.
3. Return decoded video frames and per-object predicted point clouds in the shape accepted by
   `training.pc_visualization.save_pointcloud_comparison_mp4`.
4. Run `pytest tests/test_joint_pc_pipeline.py`.

### 6. Assemble the trainer, validation sampling, and artifacts

**Files:**
- Create `train_joint_wan_physctrl.py`
- Create `tests/test_train_joint_wan_physctrl.py`
- Modify `README.md` or a focused training document only if the repository's documentation
  convention warrants a new launch example

1. Write failing lightweight tests for configuration loading, frozen VAE/text encoder parameters,
   a single optimizer/param group containing Wan DiT + PhysCtrl + bridge trainables, empty prompt
   construction, exact loss composition, and every-250-step validation trigger.
2. Implement the trainer by reusing existing I2V checkpoint/latent preparation and PC DDPM helpers.
   Load the base checkpoint; instantiate the PhysCtrl and bridge from scratch; save/load complete
   joint checkpoint state (models, optimizer, scheduler, global step).
3. On validation, call the joint pipeline and write `video.mp4` plus
   `object_000_trajectory_comparison.mp4`, etc. Call
   `training.pc_visualization.save_pointcloud_comparison_mp4` directly so the object trajectory
   render is exactly the same one used by `train_pc.py`.
4. Log `video_loss`, each object loss, `pc_loss_sum`, total loss, and bridge gradient norm.
5. Run `pytest tests/test_train_joint_wan_physctrl.py`.

### 7. Verify the integrated feature

**Files:** all files above

1. Run the focused joint suite:
   `pytest tests/test_joint_dataset.py tests/test_joint_objectives.py tests/test_joint_bridge.py tests/test_joint_pc_pipeline.py tests/test_train_joint_wan_physctrl.py`.
2. Run the legacy regression suite relevant to touched code:
   `pytest tests/test_pc_trajectory_model.py tests/test_train_pc.py tests/test_train_i2v_832x480.py`.
3. Run syntax/import checks for the new trainer and config. If GPU resources are available, perform
   one single-clip, one-object smoke forward/backward and one two-object smoke forward, without
   claiming a full training run.
4. Review `git diff` to ensure the existing unrelated dirty changes remain untouched, then report
   commands and outcomes.

## Notes for the implementer

- The design spec is `docs/superpowers/specs/2026-07-31-joint-wan-physctrl-training-design.md`.
- The new implementation should be self-contained and must not alter current standalone I2V or PC
  training semantics.
- The metadata file name is deliberately `metadata.json`, not `camera.json`.
- Do not quietly substitute an averaged PC loss, a shared video/PC timestep, a pretrained PC
  branch, a prompt, pooled bridge tokens, object-to-object attention, or separate optimizers.
