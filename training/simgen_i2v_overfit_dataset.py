"""Native RGB SimGen dataset for the I2V overfit experiment."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_TEMPLATE = "{frame:08d}.png"


class SimGenI2VOverfitDataset(Dataset):
    """Load fixed 49-frame RGB videos from ordered SimGen sample directories."""

    def __init__(
        self,
        sample_root: str | Path,
        prompt: str,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (480, 480),
        num_samples: int = 1,
    ) -> None:
        self.sample_root = Path(sample_root)
        self.prompt = prompt
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.sample_roots = self._sample_roots(num_samples)
        for root in self.sample_roots:
            self._validate_sequence(root)

    def _sample_roots(self, num_samples: int) -> list[Path]:
        if self.sample_root.name == "view_0":
            if num_samples != 1 or self.sample_root.parent.name != "sample_0":
                raise ValueError(
                    f"{self.sample_root}: expected the view directory below sample_0"
                )
            return [self.sample_root]
        return [
            self.sample_root / f"sample_{sample}" / "view_0"
            for sample in range(num_samples)
        ]

    def _validate_sequence(self, sample_root: Path) -> None:
        if not sample_root.is_dir():
            raise ValueError(f"Sample root does not exist: {sample_root}")
        if sample_root.name != "view_0" or not sample_root.parent.name.startswith("sample_"):
            raise ValueError(
                f"{sample_root}: expected a view_0 directory below a sample directory"
            )

        expected_names = {
            FRAME_TEMPLATE.format(frame=frame) for frame in range(self.expected_frames)
        }
        actual_names = {path.name for path in sample_root.glob("*.png")}
        unexpected_names = sorted(actual_names - expected_names)
        if unexpected_names:
            raise ValueError(
                f"{sample_root}: unexpected PNG frames: {unexpected_names}"
            )

        for frame in range(self.expected_frames):
            path = sample_root / FRAME_TEMPLATE.format(frame=frame)
            if not path.is_file():
                raise ValueError(f"{sample_root}: missing required frame {path.name}")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"{path}: expected RGB PNG, got mode {image.mode}")
                if image.size != self.expected_size:
                    width, height = self.expected_size
                    raise ValueError(
                        f"{path}: expected {width}x{height}, "
                        f"got {image.size[0]}x{image.size[1]}"
                    )

    def __len__(self) -> int:
        return len(self.sample_roots)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_root = self.sample_roots[index]
        frames = [
            self._load_rgb(sample_root / FRAME_TEMPLATE.format(frame=frame))
            for frame in range(self.expected_frames)
        ]
        return {
            "video": torch.stack(frames),
            "prompt": self.prompt,
            "sample_id": sample_root.parent.name,
        }

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).mul_(2.0).sub_(1.0)
