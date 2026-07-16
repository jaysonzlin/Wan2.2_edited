# VAE Latent Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `compare_vae_latent.py`, which reconstructs an RGBA PNG video sequence with Wan2.2 VAE and writes a labeled GT-versus-reconstruction MP4.

**Architecture:** Keep the executable script composed of small pure helpers for input discovery, RGB preparation, metric calculation, and comparison-frame rendering. Isolate the one GPU/checkpoint boundary in `reconstruct_video` and `main`, so pytest can verify the rest with generated RGBA fixtures and a deterministic fake VAE.

**Tech Stack:** Python, PyTorch, Pillow, imageio/FFmpeg, pytest, `wan.modules.vae2_2.Wan2_2_VAE`.

## Global Constraints

- Default `--sample-path` is `examples/sample_0`; frames are `rgba_*.png` and must be numeric-sorted by suffix.
- Default `--vae-checkpoint` is `./Wan2.2-TI2V-5B/Wan2.2_VAE.pth`; an override is supported.
- RGBA input is composited over black and converted to RGB `float32` in `[-1,1]` with shape `[3,T,H,W]`.
- Default `--fps` is 24 and default `--output` is `output/vae_comparison.mp4`.
- Output layout is GT-left and reconstruction-right, with visible labels.
- CPU tests must not load checkpoints, allocate CUDA tensors, or write a real MP4.
- Do not add the user-provided `examples/sample_0/` files to Git.

---

## File structure

- Create: `compare_vae_latent.py` — CLI and pure video-comparison helpers.
- Create: `tests/test_compare_vae_latent.py` — CPU-only behavioral coverage for all helpers.
- Reference: `wan/modules/vae2_2.py:888-1048` — `Wan2_2_VAE` public wrapper API.
- Reference: `wan/utils/utils.py:90-120` — repository’s existing imageio H.264 writer conventions.
- Reference: `docs/superpowers/specs/2026-07-16-vae-latent-comparison-design.md` — approved requirements.

### Task 1: Test input discovery and RGBA preparation

**Files:**
- Create: `tests/test_compare_vae_latent.py`
- Create: `compare_vae_latent.py`

**Interfaces:**
- Produces `discover_rgba_frames(sample_path: Path) -> list[Path]` and `load_rgb_video(frame_paths: Sequence[Path]) -> torch.Tensor`.
- `discover_rgba_frames` returns paths ordered by the integer in `rgba_<integer>.png`.
- `load_rgb_video` returns CPU `torch.float32` shape `[3,T,H,W]` in `[-1,1]`.

- [ ] **Step 1: Write the failing discovery and alpha-compositing tests**

```python
from pathlib import Path

import pytest
import torch
from PIL import Image

from compare_vae_latent import discover_rgba_frames, load_rgb_video


def write_rgba(path: Path, rgba: tuple[int, int, int, int]) -> None:
    Image.new("RGBA", (2, 1), rgba).save(path)


def test_discovers_rgba_frames_in_numeric_order(tmp_path: Path) -> None:
    write_rgba(tmp_path / "rgba_00010.png", (255, 0, 0, 255))
    write_rgba(tmp_path / "rgba_00002.png", (0, 255, 0, 255))
    write_rgba(tmp_path / "rgba_00001.png", (0, 0, 255, 255))

    frames = discover_rgba_frames(tmp_path)

    assert [path.name for path in frames] == [
        "rgba_00001.png", "rgba_00002.png", "rgba_00010.png"
    ]


def test_load_rgb_video_composites_alpha_over_black(tmp_path: Path) -> None:
    frame = tmp_path / "rgba_00000.png"
    write_rgba(frame, (255, 0, 0, 128))

    video = load_rgb_video([frame])

    assert video.shape == (3, 1, 1, 2)
    assert video.dtype == torch.float32
    assert torch.allclose(video[:, 0, 0, 0], torch.tensor([0.0039, -1.0, -1.0]), atol=0.01)


def test_discovery_rejects_directory_without_rgba_frames(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No rgba PNG frames"):
        discover_rgba_frames(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail because the module is missing**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'compare_vae_latent'`.

- [ ] **Step 3: Implement only input-discovery and loading helpers**

```python
def discover_rgba_frames(sample_path: Path) -> list[Path]:
    frames = list(sample_path.glob("rgba_*.png"))
    if not frames:
        raise FileNotFoundError(f"No rgba PNG frames found in {sample_path}")
    try:
        return sorted(frames, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    except ValueError as error:
        raise ValueError("RGBA frame names must end in an integer index") from error


def load_rgb_video(frame_paths: Sequence[Path]) -> torch.Tensor:
    rgb_frames = []
    expected_size = None
    for path in frame_paths:
        image = Image.open(path)
        if image.mode != "RGBA":
            raise ValueError(f"Expected RGBA PNG: {path}")
        if expected_size is None:
            expected_size = image.size
        elif image.size != expected_size:
            raise ValueError("All RGBA frames must have identical dimensions")
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]
        rgb_frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(rgb_frames, dim=1).mul_(2.0).sub_(1.0)
```

- [ ] **Step 4: Run tests to verify input helpers pass**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: the three Task 1 tests pass.

### Task 2: Test reconstruction, metrics, and comparison rendering

**Files:**
- Modify: `tests/test_compare_vae_latent.py`
- Modify: `compare_vae_latent.py`

**Interfaces:**
- Produces `reconstruct_video(vae, video) -> torch.Tensor`, `compute_metrics(gt, reconstruction) -> tuple[float, float]`, and `make_comparison_frame(gt_rgb, reconstruction_rgb) -> np.ndarray`.
- `reconstruct_video` calls the VAE’s list-based `encode`/`decode` interface once each.
- `make_comparison_frame` consumes two equal-size `uint8` HWC RGB arrays and returns one `uint8` HWC RGB array twice the input width.

- [ ] **Step 1: Write failing reconstruction, metric, and layout tests**

```python
import math
import numpy as np

from compare_vae_latent import compute_metrics, make_comparison_frame, reconstruct_video


class FakeVAE:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, videos):
        self.encode_calls += 1
        return videos

    def decode(self, latents):
        self.decode_calls += 1
        return [latents[0] * 0.5]


def test_reconstruct_video_uses_list_based_vae_api() -> None:
    video = torch.ones((3, 1, 2, 2))
    vae = FakeVAE()

    reconstruction = reconstruct_video(vae, video)

    assert vae.encode_calls == 1
    assert vae.decode_calls == 1
    assert torch.equal(reconstruction, video * 0.5)


def test_metrics_are_zero_mse_and_infinite_psnr_for_identical_videos() -> None:
    video = torch.zeros((3, 1, 2, 2))

    mse, psnr = compute_metrics(video, video)

    assert mse == 0.0
    assert math.isinf(psnr)


def test_comparison_frame_places_gt_left_and_reconstruction_right() -> None:
    gt = np.full((4, 5, 3), 10, dtype=np.uint8)
    reconstruction = np.full((4, 5, 3), 200, dtype=np.uint8)

    comparison = make_comparison_frame(gt, reconstruction)

    assert comparison.shape == (4, 10, 3)
    assert np.array_equal(comparison[-1, 0], [10, 10, 10])
    assert np.array_equal(comparison[-1, -1], [200, 200, 200])
```

- [ ] **Step 2: Run tests to verify the missing helpers fail**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: import fails because the three Task 2 helpers do not exist.

- [ ] **Step 3: Implement the minimal reconstruction, metric, and rendering helpers**

```python
def reconstruct_video(vae: Any, video: torch.Tensor) -> torch.Tensor:
    latents = vae.encode([video])
    return vae.decode(latents)[0]


def compute_metrics(gt: torch.Tensor, reconstruction: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((gt.float() - reconstruction.float()).square()).item()
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse / 4.0)
    return mse, psnr


def make_comparison_frame(gt_rgb: np.ndarray, reconstruction_rgb: np.ndarray) -> np.ndarray:
    if gt_rgb.shape != reconstruction_rgb.shape:
        raise ValueError("GT and reconstruction frame shapes must match")
    comparison = np.concatenate([gt_rgb, reconstruction_rgb], axis=1)
    image = Image.fromarray(comparison)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "Ground truth", fill="white")
    draw.text((gt_rgb.shape[1] + 8, 8), "VAE reconstruction", fill="white")
    return np.asarray(image)
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: all Task 1 and Task 2 tests pass.

### Task 3: Add CLI orchestration and MP4 writer

**Files:**
- Modify: `tests/test_compare_vae_latent.py`
- Modify: `compare_vae_latent.py`

**Interfaces:**
- Produces `to_uint8_frames(video) -> Iterator[np.ndarray]`, `write_comparison_video(gt, reconstruction, output, fps) -> None`, `parse_args() -> argparse.Namespace`, and `main() -> None`.
- `write_comparison_video` accepts `[3,T,H,W]` tensors in `[-1,1]`, writes H.264 at `fps`, and closes the writer in `finally`.

- [ ] **Step 1: Write failing CLI-default and writer tests**

```python
from unittest.mock import Mock, patch

from compare_vae_latent import parse_args, write_comparison_video


def test_cli_defaults_match_sample_and_checkpoint_paths(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["compare_vae_latent.py"])

    args = parse_args()

    assert args.sample_path == Path("examples/sample_0")
    assert args.vae_checkpoint == Path("Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    assert args.output == Path("output/vae_comparison.mp4")
    assert args.fps == 24


def test_writer_appends_one_frame_per_video_timestep(tmp_path: Path) -> None:
    gt = torch.zeros((3, 2, 2, 2))
    reconstruction = torch.zeros((3, 2, 2, 2))
    writer = Mock()
    with patch("compare_vae_latent.imageio.get_writer", return_value=writer):
        write_comparison_video(gt, reconstruction, tmp_path / "comparison.mp4", 24)

    assert writer.append_data.call_count == 2
    writer.close.assert_called_once()
```

- [ ] **Step 2: Run tests to verify missing orchestration helpers fail**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: import fails because `parse_args` and `write_comparison_video` do not exist.

- [ ] **Step 3: Implement CLI, safe writer lifecycle, and main**

```python
def to_uint8_frames(video: torch.Tensor) -> Iterator[np.ndarray]:
    video_uint8 = video.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    for frame in video_uint8.permute(1, 2, 3, 0):
        yield frame.numpy()


def write_comparison_video(gt, reconstruction, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8)
    try:
        for gt_frame, reconstruction_frame in zip(to_uint8_frames(gt), to_uint8_frames(reconstruction)):
            writer.append_data(make_comparison_frame(gt_frame, reconstruction_frame))
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    if not args.vae_checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    frames = discover_rgba_frames(args.sample_path)
    video = load_rgb_video(frames).to("cuda")
    vae = Wan2_2_VAE(vae_pth=str(args.vae_checkpoint), device="cuda")
    reconstruction = reconstruct_video(vae, video)
    mse, psnr = compute_metrics(video, reconstruction)
    write_comparison_video(video, reconstruction, args.output, args.fps)
    print(f"Wrote {args.output}; MSE={mse:.8f}; PSNR={psnr:.2f} dB")
```

- [ ] **Step 4: Run the full CPU test file**

Run:

```bash
python -m pytest tests/test_compare_vae_latent.py -q
```

Expected: all tests pass without CUDA or checkpoints.

- [ ] **Step 5: Run static syntax verification**

Run:

```bash
python -m py_compile compare_vae_latent.py
```

Expected: exits with code 0 and produces no output.

### Task 4: Run the real VAE comparison manually

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes the CLI from Task 3 and a locally available 5B VAE checkpoint.
- Produces `output/vae_comparison.mp4` and printed MSE/PSNR.

- [ ] **Step 1: Run the sample command on a CUDA-capable host**

```bash
python compare_vae_latent.py
```

Expected: the script reads 49 frames from `examples/sample_0`, writes `output/vae_comparison.mp4` at 24 fps, and prints finite MSE/PSNR.

- [ ] **Step 2: Inspect output properties**

Run:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 output/vae_comparison.mp4
```

Expected: width `2560`, height `704`, frame rate `24/1`, and 49 video frames.

- [ ] **Step 3: Inspect visual ordering**

Open `output/vae_comparison.mp4` and verify that the left panel is the black-composited source, the right panel is the VAE reconstruction, and labels are legible without obscuring the bottom corners used by tests.

## Plan self-review

- **Spec coverage:** Tasks 1–3 implement every CLI, RGBA, VAE, metric, layout, and writer requirement; Task 4 verifies the real CUDA/checkpoint path.
- **Evidence discipline:** Unit tests exercise public helper behavior with generated input and a fake VAE; the real VAE path is explicitly manual.
- **Scope:** no diffusion, training, VAE modifications, extra videos, or Git changes to `examples/sample_0/`.
