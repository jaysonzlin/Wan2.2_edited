# SimGen I2V History Overfit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add a one-H200, 10,000-step Wan 2.2 video-only overfit experiment that trains on native 480x480 SimGen sample_0 while conditioning on four latent-time history slots.

**Architecture:** A strict one-sample RGB dataset reads the SimGen frame layout directly. A dedicated trainer follows the established I2V training flow but uses the native dataset and pins four latent slots in training and visualization. A dedicated config and Slurm launcher leave existing I2V and joint paths unchanged.

**Tech Stack:** Python, PyTorch, Accelerate, Wan 2.2 TI2V-5B, Pillow, PyYAML, pytest, Slurm, Singularity.

**Spec:** docs/superpowers/specs/2026-09-01-simgen-i2v-history-overfit-design.md

## Global Constraints

- Load exactly sample_0/view_0/00000000.png through 00000048.png as 480x480 RGB PNGs.
- Do not use point clouds, Utonia features, joint objectives, or validation code.
- Start from Wan2.2-TI2V-5B; train the DiT and keep VAE/T5 frozen.
- Treat four latent-time slots as clean history; loss applies only to later slots.
- Resample training flow noise/timesteps each update; all visualizations use one fixed seed.
- Run 10,000 optimizer steps on one H200; visualize and checkpoint every 500; retain two checkpoints.
- Write one decoded target.mp4 from the clean sample beside the fixed-seed generated visualizations.
- Use unique output/W&B names, resume latest on requeue, disable validation, and do not create final_dit.
- Do not modify existing train_i2v.py, train_i2v_832x480.py, joint_simgen.py, or their launchers.

---

### Task 1: Add a strict native SimGen RGB dataset

**Files:**
- Create: training/simgen_i2v_overfit_dataset.py
- Create: tests/test_simgen_i2v_overfit_dataset.py

**Interfaces:**
- Produces SimGenI2VOverfitDataset(sample_root: str | Path, prompt: str, expected_frames: int = 49, expected_size: tuple[int, int] = (480, 480)).
- Returns one item with video [49, 3, 480, 480], prompt, and sample_id "sample_0".
- Consumed by train_i2v_simgen_480_overfit.py.

- [ ] **Step 1: Write the failing tests**

~~~python
def test_dataset_reads_ordered_native_rgb_frames(tmp_path):
    root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(root)
    item = SimGenI2VOverfitDataset(root, "")[0]
    assert item["video"].shape == (49, 3, 480, 480)
    assert item["sample_id"] == "sample_0"
    assert item["prompt"] == ""

def test_dataset_rejects_missing_native_frame(tmp_path):
    root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(root, skip={17})
    with pytest.raises(ValueError, match="00000017.png"):
        SimGenI2VOverfitDataset(root, "")

def test_dataset_rejects_rgba_or_wrong_size(tmp_path):
    root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(root)
    Image.new("RGBA", (480, 480)).save(root / "00000000.png")
    with pytest.raises(ValueError, match="expected RGB PNG"):
        SimGenI2VOverfitDataset(root, "")
~~~

- [ ] **Step 2: Verify red**

Run: pytest tests/test_simgen_i2v_overfit_dataset.py -v

Expected: FAIL because training.simgen_i2v_overfit_dataset does not exist.

- [ ] **Step 3: Implement the dataset**

~~~python
FRAME_TEMPLATE = "{frame:08d}.png"

class SimGenI2VOverfitDataset(Dataset):
    def __init__(self, sample_root, prompt, expected_frames=49, expected_size=(480, 480)):
        self.sample_root = Path(sample_root)
        self.prompt = prompt
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self._validate_sequence()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        frames = [self._load_rgb(frame) for frame in range(self.expected_frames)]
        return {"video": torch.stack(frames), "prompt": self.prompt, "sample_id": "sample_0"}
~~~

Validate the root, all 49 numeric filenames, absence of other PNG names, RGB mode, and 480x480 size. Convert RGB to CHW float tensors in [-1, 1]. Require the parent directory to be named sample_0.

- [ ] **Step 4: Verify green**

Run: pytest tests/test_simgen_i2v_overfit_dataset.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add training/simgen_i2v_overfit_dataset.py tests/test_simgen_i2v_overfit_dataset.py
git commit -m "feat: add native SimGen I2V overfit dataset"
~~~

### Task 2: Add and test four-slot latent pinning

**Files:**
- Modify: training/wan_i2v_training.py
- Modify: tests/test_wan_i2v_training.py

**Interfaces:**
- Produces pin_history_latents(latent: Tensor, history_latents: Tensor, history_frames: int) -> Tensor.
- Inputs and output have shape [C, T, H, W].
- Consumed by dedicated visualization sampling.

- [ ] **Step 1: Write failing tests**

~~~python
def test_pin_history_latents_replaces_exact_prefix():
    noisy = torch.zeros(2, 6, 1, 1)
    clean = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1, 1)
    result = pin_history_latents(noisy, clean, history_frames=4)
    assert torch.equal(result[:, :4], clean[:, :4])
    assert torch.equal(result[:, 4:], noisy[:, 4:])

def test_pin_history_latents_requires_target_slot():
    latent = torch.zeros(2, 4, 1, 1)
    with pytest.raises(ValueError, match="leave at least one target"):
        pin_history_latents(latent, latent, history_frames=4)
~~~

- [ ] **Step 2: Verify red**

Run: pytest tests/test_wan_i2v_training.py -k pin_history_latents -v

Expected: FAIL because pin_history_latents cannot be imported.

- [ ] **Step 3: Implement the helper**

~~~python
def pin_history_latents(latent, history_latents, history_frames):
    if latent.shape != history_latents.shape or latent.ndim != 4:
        raise ValueError("latent and history_latents must have matching [C, T, H, W] shape")
    if not 0 < history_frames < latent.shape[1]:
        raise ValueError("history_frames must leave at least one target latent slot")
    result = latent.clone()
    result[:, :history_frames] = history_latents[:, :history_frames]
    return result
~~~

Leave make_flow_matching_batch as the training-side contract and call it with history_frames=4 from the new trainer.

- [ ] **Step 4: Verify green**

Run: pytest tests/test_wan_i2v_training.py -v

Expected: PASS, including existing one-slot behavior.

- [ ] **Step 5: Commit**

~~~bash
git add training/wan_i2v_training.py tests/test_wan_i2v_training.py
git commit -m "feat: add reusable I2V history latent pinning"
~~~

### Task 3: Add the dedicated native-480 trainer

**Files:**
- Create: train_i2v_simgen_480_overfit.py
- Create: tests/test_train_i2v_simgen_480_overfit.py

**Interfaces:**
- Consumes --config configs/train/overfit_simgen_i2v_480_history.yaml and positional overrides.
- Consumes SimGenI2VOverfitDataset, make_flow_matching_batch, and pin_history_latents.
- Produces checkpoint-<step>/ directories and vis/epoch_<epoch>.mp4 only.
- Consumed by submit_simgen_i2v_480_history_overfit.sh.

- [ ] **Step 1: Write failing trainer-contract tests**

~~~python
def test_help_does_not_import_remote_dependencies():
    result = subprocess.run(
        [sys.executable, "train_i2v_simgen_480_overfit.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout

def test_trainer_uses_native_dataset_and_four_slot_history():
    source = Path("train_i2v_simgen_480_overfit.py").read_text()
    assert "SimGenI2VOverfitDataset(" in source
    assert "history_frames=4" in source
    assert "pin_history_latents(" in source

def test_trainer_does_not_export_final_dit():
    source = Path("train_i2v_simgen_480_overfit.py").read_text()
    assert 'save_state(output_dir / f"checkpoint-{global_step}")' in source
    assert 'save_pretrained(output_dir / "final_dit"' not in source
    assert '"target.mp4"' in source
~~~

- [ ] **Step 2: Verify red**

Run: pytest tests/test_train_i2v_simgen_480_overfit.py -v

Expected: FAIL because the entrypoint is absent.

- [ ] **Step 3: Implement the minimal dedicated trainer**

Copy only the established training control flow from train_i2v_832x480.py: config parsing, Accelerate, frozen encoders, trainable DiT, optimizer/scheduler, W&B, checkpoint fallback, and visualization writer. Use:

~~~python
dataset = SimGenI2VOverfitDataset(data["sample_root"], data["prompt"])
dataloader = DataLoader(
    dataset, batch_size=training["train_batch_size"], shuffle=False,
    num_workers=data["dataloader_num_workers"], pin_memory=True,
)
flow = make_flow_matching_batch(
    clean_latents, generator, training["time_shift"],
    training["num_train_timesteps"], history_frames=4,
)
~~~

Make sample_visualization_latent receive the whole video and history_frames. Encode it, initialize fixed-seed noise with the same [C, T, H, W] shape, pin four slots before the solver, set their frame times to zero, and re-pin after every scheduler step:

~~~python
clean_latents = vae.encode([video.permute(1, 0, 2, 3).contiguous()])[0]
latent = torch.randn_like(clean_latents, generator=generator)
latent = pin_history_latents(latent, clean_latents, history_frames)
frame_times[:, :history_frames] = 0
...
latent = pin_history_latents(latent, clean_latents, history_frames)
~~~

Use training["visualization_seed"]. Retain checkpoint pruning but remove the final save_pretrained block.
Before the training loop, on the main process, decode and write clean_latents once as output_dir / "vis" / "target.mp4" using the same FPS as generated visualizations.

- [ ] **Step 4: Verify green**

Run: pytest tests/test_train_i2v_simgen_480_overfit.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add train_i2v_simgen_480_overfit.py tests/test_train_i2v_simgen_480_overfit.py
git commit -m "feat: add SimGen history I2V overfit trainer"
~~~

### Task 4: Add the config and one-H200 requeue launcher

**Files:**
- Create: configs/train/overfit_simgen_i2v_480_history.yaml
- Create: submit_simgen_i2v_480_history_overfit.sh
- Create: tests/test_simgen_i2v_history_overfit_config.py
- Create: tests/test_submit_simgen_i2v_480_history_overfit.py

**Interfaces:**
- Consumes the trainer and configs/accelerate/h200_single_gpu.yaml.
- Produces a self-contained experiment invocation.

- [ ] **Step 1: Write failing config and launcher tests**

~~~python
def test_config_has_native_history_overfit_settings():
    config = load_config("configs/train/overfit_simgen_i2v_480_history.yaml", [])
    assert config["data"]["sample_root"].endswith("sample_0/view_0")
    assert (config["data"]["width"], config["data"]["height"]) == (480, 480)
    assert config["training"]["max_train_steps"] == 10_000
    assert config["training"]["checkpoint_every_steps"] == 500
    assert config["training"]["visualization_every_steps"] == 500
    assert config["training"]["checkpoints_total_limit"] == 2
    assert config["validation"]["enabled"] is False

def test_submit_script_runs_one_h200_with_latest_resume():
    script = Path("submit_simgen_i2v_480_history_overfit.sh").read_text()
    assert "#SBATCH --partition=gpu_h200" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --requeue" in script
    assert "configs/accelerate/h200_single_gpu.yaml" in script
    assert "train_i2v_simgen_480_overfit.py" in script
    assert "training.resume_from_checkpoint=latest" in script
~~~

- [ ] **Step 2: Verify red**

Run: pytest tests/test_simgen_i2v_history_overfit_config.py tests/test_submit_simgen_i2v_480_history_overfit.py -v

Expected: FAIL because the config and submit script are absent.

- [ ] **Step 3: Implement the config and launcher**

Base the YAML on configs/train/overfit_kubric_i2v_832x480.yaml, retaining its optimizer settings. Set:

~~~yaml
data:
  sample_root: ../simgen/runs/panda_ball_can/sample_0/view_0
  prompt: ""
  num_frames: 49
  width: 480
  height: 480
  dataloader_num_workers: 0
training:
  max_train_steps: 10000
  checkpoint_every_steps: 500
  checkpoints_total_limit: 2
  visualization_every_steps: 500
  visualization_seed: 42
  resume_from_checkpoint: null
logging:
  output_dir: outputs/simgen_i2v_480_history_overfit
  wandb_project: wan22-simgen-i2v-overfit
  wandb_run_name: simgen-sample-0-history-480
validation:
  enabled: false
~~~

Create the submit script from submit_832x480.sh. Use gpu_h200, one GPU, requeue, new log paths, current.sif, the single-GPU Accelerate config, the dedicated trainer/config, and training.resume_from_checkpoint=latest. Do not add shell overrides for the schedule or retention.

- [ ] **Step 4: Verify green and shell syntax**

Run: pytest tests/test_simgen_i2v_history_overfit_config.py tests/test_submit_simgen_i2v_480_history_overfit.py -v && bash -n submit_simgen_i2v_480_history_overfit.sh

Expected: PASS and exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add configs/train/overfit_simgen_i2v_480_history.yaml submit_simgen_i2v_480_history_overfit.sh tests/test_simgen_i2v_history_overfit_config.py tests/test_submit_simgen_i2v_480_history_overfit.py
git commit -m "feat: add SimGen I2V history overfit launch"
~~~

### Task 5: Verify the integrated experiment surface

**Files:**
- Verify: all Task 1–4 files.

**Interfaces:**
- Consumes every previous deliverable.
- Produces evidence of structural correctness without GPU execution.

- [ ] **Step 1: Run focused tests**

Run: pytest tests/test_simgen_i2v_overfit_dataset.py tests/test_wan_i2v_training.py tests/test_train_i2v_simgen_480_overfit.py tests/test_simgen_i2v_history_overfit_config.py tests/test_submit_simgen_i2v_480_history_overfit.py -v

Expected: PASS with zero failures.

- [ ] **Step 2: Run static entrypoint and shell checks**

Run: python train_i2v_simgen_480_overfit.py --help && bash -n submit_simgen_i2v_480_history_overfit.sh && git diff --check

Expected: exit 0.

- [ ] **Step 3: Inspect all requirements**

Confirm native sample_0-only loading; 480x480 RGB; four clean latent slots for train and sample; target-only loss; 10,000 steps; 500-step cadence; two checkpoint directories; no validation/final_dit; fixed visualization seed; Wan initialization; W&B; one-H200 latest-resume launcher.
