import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from train_i2v_simgen_480_overfit import (
    build_dataset,
    make_history_conditioned_visualization_latent,
)


def test_help_does_not_import_remote_only_training_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "train_i2v_simgen_480_overfit.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_visualization_latent_pins_four_history_slots() -> None:
    clean_latents = torch.zeros(2, 6, 1, 1)

    latent = make_history_conditioned_visualization_latent(
        clean_latents, torch.Generator().manual_seed(42), history_frames=4
    )

    assert latent.shape == clean_latents.shape
    assert torch.equal(latent[:, :4], clean_latents[:, :4])
    assert not torch.equal(latent[:, 4:], clean_latents[:, 4:])


def _make_rgb_sequence(root: Path) -> None:
    root.mkdir(parents=True)
    for frame in range(49):
        Image.new("RGB", (480, 480)).save(root / f"{frame:08d}.png")


def test_build_dataset_uses_configured_sample_count(tmp_path: Path) -> None:
    for sample in range(2):
        _make_rgb_sequence(tmp_path / f"sample_{sample}" / "view_0")

    dataset = build_dataset(
        {"sample_root": str(tmp_path), "prompt": "prompt", "num_samples": 2}
    )

    assert len(dataset) == 2
    assert dataset[1]["sample_id"] == "sample_1"
