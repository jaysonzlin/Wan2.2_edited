"""Compare RGB video frames with their Wan2.2 VAE reconstruction."""

import argparse
import math
from pathlib import Path
import re
from typing import Any, Sequence

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


_RGBA_FRAME_PATTERN = re.compile(r"rgba_(\d+)$")


def discover_rgba_frames(sample_path: Path) -> list[Path]:
    """Return RGBA PNG frames ordered by their numeric filename suffix."""
    frames = list(sample_path.glob("rgba_*.png"))
    if not frames:
        raise FileNotFoundError(f"No rgba PNG frames found in {sample_path}")

    indexed_frames = []
    for frame in frames:
        match = _RGBA_FRAME_PATTERN.fullmatch(frame.stem)
        if match is None:
            raise ValueError(f"RGBA frame name must end in an integer index: {frame}")
        indexed_frames.append((int(match.group(1)), frame))
    return [frame for _, frame in sorted(indexed_frames)]


def load_rgb_video(frame_paths: Sequence[Path]) -> torch.Tensor:
    """Load RGBA frames over black as an RGB video tensor in [-1, 1]."""
    if not frame_paths:
        raise ValueError("At least one RGBA frame is required")

    rgb_frames = []
    expected_size = None
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            if image.mode != "RGBA":
                raise ValueError(f"Expected RGBA PNG: {frame_path}")
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ValueError("All RGBA frames must have identical dimensions")

            rgba = np.array(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]
        rgb_frames.append(torch.from_numpy(rgb).permute(2, 0, 1))

    return torch.stack(rgb_frames, dim=1).mul_(2.0).sub_(1.0)


def reconstruct_video(vae: Any, video: torch.Tensor) -> torch.Tensor:
    """Run the Wan wrapper's list-based encode/decode API for one video."""
    latents = vae.encode([video])
    return vae.decode(latents)[0]


def compute_metrics(
    ground_truth: torch.Tensor, reconstruction: torch.Tensor
) -> tuple[float, float]:
    """Return RGB-space MSE and PSNR for tensors whose range is [-1, 1]."""
    if ground_truth.shape != reconstruction.shape:
        raise ValueError("Ground truth and reconstruction video shapes must match")
    mse = torch.mean((ground_truth.float() - reconstruction.float()).square()).item()
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse / 4.0)
    return mse, psnr


def make_comparison_frame(
    ground_truth: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """Place equal-size uint8 RGB frames side by side with fixed labels."""
    if ground_truth.shape != reconstruction.shape:
        raise ValueError("GT and reconstruction frame shapes must match")
    comparison = np.concatenate([ground_truth, reconstruction], axis=1)
    image = Image.fromarray(comparison)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "Ground truth", fill="white")
    draw.text((ground_truth.shape[1] + 8, 8), "VAE reconstruction", fill="white")
    return np.asarray(image)


def to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    """Convert a [C, T, H, W] video in [-1, 1] to uint8 RGB frames."""
    uint8_video = (
        video.detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
    )
    return [frame.numpy() for frame in uint8_video.permute(1, 2, 3, 0)]


def write_comparison_video(
    ground_truth: torch.Tensor,
    reconstruction: torch.Tensor,
    output_path: Path,
    fps: int,
) -> None:
    """Write an H.264 side-by-side GT/reconstruction comparison video."""
    if ground_truth.shape != reconstruction.shape:
        raise ValueError("Ground truth and reconstruction video shapes must match")
    if fps <= 0:
        raise ValueError("FPS must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    try:
        for ground_truth_frame, reconstruction_frame in zip(
            to_uint8_frames(ground_truth), to_uint8_frames(reconstruction), strict=True
        ):
            writer.append_data(
                make_comparison_frame(ground_truth_frame, reconstruction_frame)
            )
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    """Parse standalone comparison-script options."""
    parser = argparse.ArgumentParser(
        description="Encode and decode sample RGBA frames with the Wan2.2 VAE."
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=Path("examples/sample_0"),
        help="Directory containing rgba_*.png video frames.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path("Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        help="Path to Wan2.2_VAE.pth (the README-prescribed generate.py layout).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/vae_comparison.mp4"),
        help="Comparison MP4 output path.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    return parser.parse_args()


def main() -> None:
    """Load the VAE on CUDA, reconstruct one sample, and write its comparison."""
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("FPS must be positive")
    if not args.vae_checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 VAE comparison requires a CUDA-capable device")

    from wan.modules.vae2_2 import Wan2_2_VAE

    frame_paths = discover_rgba_frames(args.sample_path)
    ground_truth = load_rgb_video(frame_paths).to("cuda")
    vae = Wan2_2_VAE(vae_pth=str(args.vae_checkpoint), device="cuda")

    with torch.no_grad():
        reconstruction = reconstruct_video(vae, ground_truth)

    mse, psnr = compute_metrics(ground_truth, reconstruction)
    write_comparison_video(ground_truth, reconstruction, args.output, args.fps)
    psnr_text = "inf" if math.isinf(psnr) else f"{psnr:.4f} dB"
    print(f"MSE: {mse:.8f}")
    print(f"PSNR: {psnr_text}")
    print(f"Wrote comparison video: {args.output}")


if __name__ == "__main__":
    main()
