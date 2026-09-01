from pathlib import Path

import pytest
import torch
from PIL import Image

from training.simgen_i2v_overfit_dataset import SimGenI2VOverfitDataset


def make_rgb_sequence(root: Path, *, skip: set[int] | None = None) -> None:
    root.mkdir(parents=True)
    skipped = skip or set()
    for frame in range(49):
        if frame not in skipped:
            Image.new("RGB", (480, 480), (frame, 0, 0)).save(
                root / f"{frame:08d}.png"
            )


def test_dataset_reads_ordered_native_rgb_frames(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(sample_root)

    item = SimGenI2VOverfitDataset(sample_root, "")[0]

    assert item["video"].shape == (49, 3, 480, 480)
    assert torch.equal(item["video"][0, :, 0, 0], torch.full((3,), -1.0))
    assert item["sample_id"] == "sample_0"
    assert item["prompt"] == ""


def test_dataset_rejects_missing_native_frame(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(sample_root, skip={17})

    with pytest.raises(ValueError, match="00000017.png"):
        SimGenI2VOverfitDataset(sample_root, "")


def test_dataset_rejects_non_rgb_native_frame(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_0" / "view_0"
    make_rgb_sequence(sample_root)
    Image.new("RGBA", (480, 480)).save(sample_root / "00000000.png")

    with pytest.raises(ValueError, match="expected RGB PNG"):
        SimGenI2VOverfitDataset(sample_root, "")
