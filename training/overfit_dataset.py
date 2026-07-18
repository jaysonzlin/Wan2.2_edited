"""Strict RGB video dataset for the Kubric I2V overfit run."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_TEMPLATE = "rgba_{frame:05d}.png"


class KubricI2VOverfitDataset(Dataset):
    """Load complete, validated Kubric RGBA sequences as RGB video tensors."""

    def __init__(
        self,
        dataset_root: str | Path,
        prompt: str,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (1280, 704),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.prompt = prompt
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self.sample_dirs = self._discover_and_validate_samples()

    def _discover_and_validate_samples(self) -> list[Path]:
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")

        sample_dirs = sorted(
            path for path in self.dataset_root.glob("sample_*") if path.is_dir()
        )
        if not sample_dirs:
            raise ValueError(f"No sample_* directories found in {self.dataset_root}")

        for sample_dir in sample_dirs:
            self._validate_sample(sample_dir)
        return sample_dirs

    def _validate_sample(self, sample_dir: Path) -> None:
        for frame in range(self.expected_frames):
            path = sample_dir / FRAME_TEMPLATE.format(frame=frame)
            if not path.is_file():
                raise ValueError(f"{sample_dir}: missing required frame {path.name}")
            with Image.open(path) as image:
                if image.mode != "RGBA":
                    raise ValueError(f"{path}: expected RGBA PNG, got mode {image.mode}")
                if image.size != self.expected_size:
                    raise ValueError(
                        f"{path}: expected {self.expected_size[0]}x{self.expected_size[1]}, "
                        f"got {image.size[0]}x{image.size[1]}"
                    )

        expected_names = {
            FRAME_TEMPLATE.format(frame=frame) for frame in range(self.expected_frames)
        }
        rgba_names = {path.name for path in sample_dir.glob("rgba_*.png")}
        unexpected_names = sorted(rgba_names - expected_names)
        if unexpected_names:
            raise ValueError(f"{sample_dir}: unexpected RGBA frames: {unexpected_names}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_dir = self.sample_dirs[index]
        frames = [
            self._load_frame(sample_dir / FRAME_TEMPLATE.format(frame=frame))
            for frame in range(self.expected_frames)
        ]
        return {
            "video": torch.stack(frames),
            "prompt": self.prompt,
            "sample_id": sample_dir.name,
        }

    def _load_frame(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).mul_(2.0).sub_(1.0)
