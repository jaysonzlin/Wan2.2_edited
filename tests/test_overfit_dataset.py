from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from training.overfit_dataset import KubricI2VOverfitDataset


def make_rgba_sequence(root: Path, alpha_for_frame_zero: int = 255, skip=None):
    root.mkdir(parents=True)
    skipped = set() if skip is None else set(skip)
    for frame in range(49):
        if frame in skipped:
            continue
        alpha = alpha_for_frame_zero if frame == 0 else 255
        Image.new("RGBA", (1280, 704), (255, 0, 0, alpha)).save(
            root / f"rgba_{frame:05d}.png"
        )


class KubricI2VOverfitDatasetTests(unittest.TestCase):
    def test_dataset_composites_alpha_over_black_and_orders_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            make_rgba_sequence(tmp_path / "sample_0", alpha_for_frame_zero=0)

            item = KubricI2VOverfitDataset(
                tmp_path, "Objects moving in a Kubric simulator"
            )[0]

            self.assertEqual(item["video"].shape, (49, 3, 704, 1280))
            self.assertTrue(torch.equal(item["video"][0, :, 0, 0], torch.full((3,), -1.0)))
            self.assertTrue(
                torch.equal(item["video"][1, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
            )
            self.assertEqual(item["prompt"], "Objects moving in a Kubric simulator")
            self.assertEqual(item["sample_id"], "sample_0")

    def test_dataset_rejects_missing_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            make_rgba_sequence(tmp_path / "sample_0", skip={17})

            with self.assertRaisesRegex(ValueError, "rgba_00017.png"):
                KubricI2VOverfitDataset(tmp_path, "prompt")
