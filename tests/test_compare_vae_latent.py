from pathlib import Path
import math
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from compare_vae_latent import (
    compute_metrics,
    discover_rgba_frames,
    load_rgb_video,
    make_comparison_frame,
    parse_args,
    reconstruct_video,
    write_comparison_video,
)


def write_rgba(path: Path, rgba: tuple[int, int, int, int]) -> None:
    Image.new("RGBA", (2, 1), rgba).save(path)


def test_discovers_rgba_frames_in_numeric_order(tmp_path: Path) -> None:
    write_rgba(tmp_path / "rgba_00010.png", (255, 0, 0, 255))
    write_rgba(tmp_path / "rgba_00002.png", (0, 255, 0, 255))
    write_rgba(tmp_path / "rgba_00001.png", (0, 0, 255, 255))

    frames = discover_rgba_frames(tmp_path)

    assert [path.name for path in frames] == [
        "rgba_00001.png",
        "rgba_00002.png",
        "rgba_00010.png",
    ]


def test_load_rgb_video_composites_alpha_over_black(tmp_path: Path) -> None:
    frame = tmp_path / "rgba_00000.png"
    write_rgba(frame, (255, 0, 0, 128))

    video = load_rgb_video([frame])

    assert video.shape == (3, 1, 1, 2)
    assert video.dtype == torch.float32
    assert torch.allclose(
        video[:, 0, 0, 0],
        torch.tensor([0.0039, -1.0, -1.0]),
        atol=0.01,
    )


def test_discovery_rejects_directory_without_rgba_frames(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No rgba PNG frames"):
        discover_rgba_frames(tmp_path)


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
    gt = np.full((32, 5, 3), 10, dtype=np.uint8)
    reconstruction = np.full((32, 5, 3), 200, dtype=np.uint8)

    comparison = make_comparison_frame(gt, reconstruction)

    assert comparison.shape == (32, 10, 3)
    assert np.array_equal(comparison[-1, 0], [10, 10, 10])
    assert np.array_equal(comparison[-1, -1], [200, 200, 200])


def test_cli_defaults_match_sample_and_checkpoint_paths(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["compare_vae_latent.py"])

    args = parse_args()

    assert args.sample_path == Path("examples/sample_0")
    assert args.vae_checkpoint == Path("Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    assert args.output == Path("output/vae_comparison.mp4")
    assert args.fps == 24


def test_cli_accepts_sample_path_override(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_vae_latent.py",
            "--sample-path",
            "custom/sample",
            "--vae-checkpoint",
            "/tmp/vae.pth",
            "--output",
            "result.mp4",
            "--fps",
            "12",
        ],
    )

    args = parse_args()

    assert args.sample_path == Path("custom/sample")
    assert args.vae_checkpoint == Path("/tmp/vae.pth")
    assert args.output == Path("result.mp4")
    assert args.fps == 12


def test_writer_appends_one_frame_per_video_timestep(tmp_path: Path) -> None:
    ground_truth = torch.zeros((3, 2, 32, 32))
    reconstruction = torch.zeros((3, 2, 32, 32))
    writer = Mock()
    output_path = tmp_path / "comparison.mp4"

    with patch("compare_vae_latent.imageio.get_writer", return_value=writer) as get_writer:
        write_comparison_video(
            ground_truth,
            reconstruction,
            output_path,
            24,
        )

    get_writer.assert_called_once_with(
        str(output_path),
        fps=24,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    assert writer.append_data.call_count == 2
    writer.close.assert_called_once()
