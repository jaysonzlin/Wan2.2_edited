"""Video-only native SimGen samples for point-conditioning ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def simgen_video_collate(samples: list[dict[str, object]]) -> dict[str, object]:
    """Collate the batch-size-one video-only training route."""
    if len(samples) != 1:
        raise ValueError("simgen_video_collate supports train_batch_size=1 only")
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in samples[0].items()
    }


class SimGenVideoDataset(Dataset):
    """Read native SimGen RGB frames without accessing object-side inputs."""

    def __init__(
        self,
        dataset_root: str | Path,
        sample_ids: list[int],
        *,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (480, 480),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")
        self.sample_dirs = [
            self.dataset_root / f"sample_{sample_id}" for sample_id in sample_ids
        ]
        missing = [path for path in self.sample_dirs if not path.is_dir()]
        if missing:
            raise ValueError(f"Missing configured SimGen sample: {missing[0]}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_dir = self.sample_dirs[index]
        frames = [
            self._load_frame(sample_dir / "view_0" / f"{frame:08d}.png")
            for frame in range(self.expected_frames)
        ]
        return {"video": torch.stack(frames), "sample_id": sample_dir.name}

    def _load_frame(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise ValueError(f"{path.parent.parent}: missing required frame {path.name}")
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"{path}: expected RGB PNG, got mode {image.mode}")
            if image.size != self.expected_size:
                raise ValueError(
                    f"{path}: expected {self.expected_size[0]}x{self.expected_size[1]}, "
                    f"got {image.size[0]}x{image.size[1]}"
                )
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).mul_(2.0).sub_(1.0)
