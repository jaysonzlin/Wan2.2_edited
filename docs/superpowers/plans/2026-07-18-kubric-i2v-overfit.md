# Kubric I2V Overfit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add a reproducible Accelerate training workflow that overfits Wan2.2-TI2V-5B on the local Kubric RGBA sequences.

**Architecture:** A strict dataset returns each complete 49-frame clip plus the fixed caption. A separate Accelerate entrypoint builds a frozen VAE/T5 and trainable TI2V DiT, applies native first-frame latent conditioning, and minimizes masked flow-matching velocity loss. CPU tests cover all custom logic; training itself runs only on the remote H200.

**Tech Stack:** Python 3.10, PyTorch CUDA 12.4, Wan2.2 TI2V, Accelerate, Transformers, OmegaConf, Pillow, imageio/ffmpeg, W&B, pytest.

## Global Constraints

- Input is exactly 49 files \`rgba_00000.png\` through \`rgba_00048.png\` in every \`training_dataset/sample_*\` directory.
- Keep native 1280x704, no augmentation, and composite RGBA over black before RGB \`[-1, 1]\` normalization.
- Use frame 0 as clean I2V condition and all remaining latent temporal positions as targets; no validation split.
- Freeze VAE and T5; train the whole TI2V DiT. Do not use Lightning, 8-bit Adam, depth, actions, proprioception, or point clouds.
- Match X-WAM defaults: AdamW \`lr=1e-5\`, \`weight_decay=0.01\`, uniform flow timestep, shift 5.0, 200 warmup steps, cosine schedule, gradient clip 1.0.
- H200 defaults: bf16, FlashAttention, gradient checkpointing, batch 1, accumulation 4, 5,000 optimizer updates.
- W&B receives scalars only. Rank zero writes \`outputs/vis/epoch_####.mp4\` every 500 updates.
- Save complete Accelerate state every 250 updates; retain latest 3; export final DiT weights.

---

### Task 1: Strict RGBA video dataset

**Files:**
- Create: \`training/__init__.py\`
- Create: \`training/overfit_dataset.py\`
- Test: \`tests/test_overfit_dataset.py\`

**Interfaces:**
- \`KubricI2VOverfitDataset(root, prompt, expected_frames=49, expected_size=(1280, 704))\`.
- \`__getitem__\` returns \`{"video": Tensor[49,3,704,1280], "prompt": str, "sample_id": str}\`.
- It raises \`ValueError\` on a missing/non-contiguous RGBA filename, wrong size, or non-RGBA source.

- [ ] **Step 1: Write the failing test**

~~~python
def test_dataset_composites_alpha_over_black_and_orders_frames(tmp_path):
    make_rgba_sequence(tmp_path / "sample_0", alpha_for_frame_zero=0)
    item = KubricI2VOverfitDataset(tmp_path, "Objects moving in a Kubric simulator")[0]
    assert item["video"].shape == (49, 3, 704, 1280)
    assert torch.equal(item["video"][0, :, 0, 0], torch.full((3,), -1.0))
    assert item["prompt"] == "Objects moving in a Kubric simulator"

def test_dataset_rejects_missing_frame(tmp_path):
    make_rgba_sequence(tmp_path / "sample_0", skip={17})
    with pytest.raises(ValueError, match="rgba_00017.png"):
        KubricI2VOverfitDataset(tmp_path, "prompt")
~~~

- [ ] **Step 2: Verify RED**

Run: \`pytest tests/test_overfit_dataset.py -v\`

Expected: FAIL because \`training.overfit_dataset\` does not exist.

- [ ] **Step 3: Implement the minimal data contract**

Use \`FRAME_TEMPLATE = "rgba_{frame:05d}.png"\`; enumerate sorted \`sample_*\` directories; check every expected path before loading; load with \`Image.open(path).convert("RGBA")\`; assert \`image.mode == "RGBA"\` before conversion and \`image.size == (1280, 704)\`; compute \`rgb = rgba[..., :3] * rgba[..., 3:4]\`; convert CHW to \`rgb * 2 - 1\`; stack frames by numeric index. Do not inspect other modality files.

- [ ] **Step 4: Verify GREEN**

Run: \`pytest tests/test_overfit_dataset.py -v\`

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add training/__init__.py training/overfit_dataset.py tests/test_overfit_dataset.py
git commit -m "feat: add strict Kubric I2V dataset"
~~~

### Task 2: Typed YAML configuration and environment assets

**Files:**
- Create: \`training/overfit_config.py\`
- Create: \`configs/train/overfit_kubric_i2v.yaml\`
- Create: \`configs/accelerate/h200_single_gpu.yaml\`
- Create: \`environment_finetune.yml\`
- Modify: \`requirements.txt\`
- Test: \`tests/test_overfit_config.py\`

**Interfaces:**
- \`load_config(path: str | Path, overrides: list[str]) -> DictConfig\`.
- CLI dot-list overrides merge after the YAML.
- Reject frame counts that are not \`4n+1\`, nonpositive intervals/steps, and a nonempty validation configuration.

- [ ] **Step 1: Write the failing test**

~~~python
def test_dotlist_overrides_win_over_yaml(tmp_path):
    config = load_config(write_yaml(tmp_path, {"training": {"max_train_steps": 5000}}),
                         ["training.max_train_steps=3"])
    assert config.training.max_train_steps == 3

def test_invalid_temporal_length_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="4n \\+ 1"):
        load_config(write_yaml(tmp_path, {"data": {"num_frames": 48}}), [])
~~~

- [ ] **Step 2: Verify RED**

Run: \`pytest tests/test_overfit_config.py -v\`

Expected: FAIL because \`training.overfit_config\` does not exist.

- [ ] **Step 3: Implement config plus static assets**

Load with \`OmegaConf.load\`, merge \`OmegaConf.from_dotlist(overrides)\`, then validate. The training YAML must set the exact fixed prompt; data root \`training_dataset\`; 49 frames; source size 1280x704; model checkpoint path \`checkpoints/Wan2.2-TI2V-5B\`; full-DiT/frozen-encoder policy; X-WAM optimizer/scheduler values; batch/accumulation/clip values; scalar, visualization and checkpoint cadences; output and W&B settings.

Set the Accelerate YAML to \`distributed_type: "NO"\`, \`num_processes: 1\`, and \`mixed_precision: "bf16"\`. Define \`wan2-2-finetune\` in \`environment_finetune.yml\` with Python 3.10, ffmpeg, pip and documented CUDA 12.4 PyTorch installation. Add commented \`# Added for Kubric I2V fine-tuning\` requirement lines for \`wandb\`, \`omegaconf\`, and Pillow only if absent. Do not add Lightning/bitsandbytes.

- [ ] **Step 4: Verify GREEN**

Run: \`pytest tests/test_overfit_config.py -v\`

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add training/overfit_config.py tests/test_overfit_config.py configs environment_finetune.yml requirements.txt
git commit -m "feat: add H200 I2V overfit configuration"
~~~

### Task 3: Wan flow-matching training helpers

**Files:**
- Create: \`training/wan_i2v_training.py\`
- Test: \`tests/test_wan_i2v_training.py\`

**Interfaces:**
- \`make_flow_matching_batch(clean_latents, generator, time_shift, num_train_timesteps)\` returns \`model_input\`, \`velocity_target\`, \`latent_timesteps\`, and \`loss_mask\`.
- \`masked_velocity_mse(prediction, target, mask) -> Tensor\`.
- \`load_frozen_encoders(checkpoint_dir, config, device)\` and \`load_trainable_dit(checkpoint_dir, config, device)\`.

- [ ] **Step 1: Write the failing test**

~~~python
def test_first_latent_slot_stays_clean_and_has_no_loss():
    clean = torch.zeros(1, 16, 13, 44, 80)
    batch = make_flow_matching_batch(clean, torch.Generator().manual_seed(0), 5.0, 1000)
    assert torch.equal(batch.model_input[:, :, :1], clean[:, :, :1])
    assert not batch.loss_mask[:, :, :1].any()
    assert batch.loss_mask[:, :, 1:].all()

def test_masked_loss_ignores_the_conditioned_slot():
    pred = torch.ones(1, 1, 2, 1, 1)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[[0]], [[1]]]]], dtype=torch.float32)
    assert masked_velocity_mse(pred, target, mask) == pytest.approx(1.0)
~~~

- [ ] **Step 2: Verify RED**

Run: \`pytest tests/test_wan_i2v_training.py -v\`

Expected: FAIL because \`training.wan_i2v_training\` does not exist.

- [ ] **Step 3: Implement exact isolated math**

Sample a uniform \`u\`; shift it with \`t = 5u / (1 + 4u)\`; sample normal noise; create \`x_t = (1-t)x_clean + t*noise\`; set velocity target to \`noise - x_clean\`; restore the clean first temporal latent in \`x_t\`; set its timestep and loss mask to zero. Expand the mask to all latent channels/spatial elements and divide MSE by mask sum.

Load the existing \`Wan2_2_VAE\`, \`T5EncoderModel\`, and \`WanModel.from_pretrained\`. Freeze the first two with \`eval().requires_grad_(False)\` and use them under \`torch.no_grad()\`. Use the trainable DiT in \`train()\`, enable its supported gradient checkpointing, and build token timesteps/sequence length according to \`wan/textimage2video.py\`. For this shape, \`seq_len = 13 * 44 * 80 / 4 = 11,440\` (not 45,760: the 2x2 spatial patch reduces token count).

- [ ] **Step 4: Verify GREEN**

Run: \`pytest tests/test_wan_i2v_training.py -v\`

Expected: PASS on CPU without a checkpoint.

- [ ] **Step 5: Commit**

~~~bash
git add training/wan_i2v_training.py tests/test_wan_i2v_training.py
git commit -m "feat: add Wan I2V flow matching helpers"
~~~

### Task 4: Accelerate training entrypoint and remote workflow

**Files:**
- Create: \`train_i2v.py\`
- Test: \`tests/test_train_i2v.py\`
- Modify: \`.gitignore\`
- Modify: \`README.md\`

**Interfaces:**
- CLI: \`python train_i2v.py --config PATH [key=value ...]\`.
- Launcher: \`accelerate launch --config_file configs/accelerate/h200_single_gpu.yaml train_i2v.py --config configs/train/overfit_kubric_i2v.yaml\`.
- \`visualization_path(output_dir, epoch) -> Path\`.
- \`prune_checkpoints(root, limit=3) -> None\`.

- [ ] **Step 1: Write the failing test**

~~~python
def test_visualization_name_uses_epoch(tmp_path):
    assert visualization_path(tmp_path, 12) == tmp_path / "vis" / "epoch_0012.mp4"

def test_checkpoint_pruning_keeps_newest_three(tmp_path):
    for step in (250, 500, 750, 1000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    prune_checkpoints(tmp_path, 3)
    assert {p.name for p in tmp_path.iterdir()} == {
        "checkpoint-500", "checkpoint-750", "checkpoint-1000"
    }
~~~

- [ ] **Step 2: Verify RED**

Run: \`pytest tests/test_train_i2v.py -v\`

Expected: FAIL because \`train_i2v\` does not exist.

- [ ] **Step 3: Implement the loop**

Build \`Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="wandb")\`, seed all processes, construct the dataset/DataLoader, and create standard AdamW plus \`get_cosine_schedule_with_warmup(optimizer, 200, 5000)\`. Pass only DiT, optimizer, scheduler, and DataLoader to \`accelerator.prepare\`.

Inside \`accelerator.accumulate\`, encode video/text without gradients, calculate masked velocity loss, backpropagate, record the preclip global norm, clip at 1.0, and step optimizer/scheduler only on synchronized updates. Log only scalar loss/LR/gradient norm through \`accelerator.log\`. Initialize W&B from YAML/environment key.

At every 250 synchronized updates, rank zero calls \`accelerator.save_state\` to \`checkpoint-{step}\` and prunes older states. At every 500 updates, rank zero uses the native TI2V sampling algorithm with \`sample_0/rgba_00000.png\`, then writes an MP4 to \`outputs/vis/epoch_####.mp4\`. Save the final unwrapped DiT state dict. Implement \`resume_from_checkpoint=latest|PATH\` via \`accelerator.load_state\` and recovered global step.

- [ ] **Step 4: Verify GREEN without CUDA**

Run: \`pytest tests/test_train_i2v.py tests/test_overfit_dataset.py tests/test_overfit_config.py tests/test_wan_i2v_training.py -v && python -m compileall train_i2v.py training\`

Expected: PASS. Do not run model loading or CUDA training on this machine.

- [ ] **Step 5: Document and ignore generated artifacts**

Add ignore entries for \`outputs/\`, \`checkpoints/\`, \`wandb/\`, and \`accelerate_state/\` without disturbing existing rules. Add README instructions for creating \`wan2-2-finetune\`, installing CUDA-12.4 PyTorch, setting \`WANDB_API_KEY\`, placing the base checkpoint, launching/resuming the job, and the H200-only limitation.

- [ ] **Step 6: Commit**

~~~bash
git add train_i2v.py tests/test_train_i2v.py .gitignore README.md
git commit -m "feat: add Accelerate Kubric I2V overfit trainer"
~~~

### Task 5: Final verification

**Files:**
- Verify: all files above.

- [ ] **Step 1: Inspect the final diff**

Run: \`git diff --check HEAD~4..HEAD && git status --short\`

Expected: no whitespace errors and no generated artifacts/dependency directories staged.

- [ ] **Step 2: Run all CPU checks**

Run: \`pytest tests/test_overfit_dataset.py tests/test_overfit_config.py tests/test_wan_i2v_training.py tests/test_train_i2v.py -v && python -m compileall train_i2v.py training\`

Expected: PASS. State clearly that CUDA execution, downloaded weights, FlashAttention, and live W&B authentication remain unverified locally because no GPU is available.

