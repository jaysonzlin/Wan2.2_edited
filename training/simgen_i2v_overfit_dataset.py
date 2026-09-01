"""Native RGB SimGen dataset for the single-sample I2V overfit experiment."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_TEMPLATE = "{frame:08d}.png"


class SimGenI2VOverfitDataset(Dataset):
    """Load the fixed 49-frame RGB video in a SimGen ``sample_0/view_0`` root."""

    def __init__(
        self,
        sample_root: str | Path,
        prompt: str,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (480, 480),
    ) -> None:
        self.sample_root = Path(sample_root)
        self.prompt = prompt
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self._validate_sequence()

    def _validate_sequence(self) -> None:
        if self.sample_root.parent.name != "sample_0":
            raise ValueError(
                f"{self.sample_root}: expected the view directory below sample_0"
            )
        if not self.sample_root.is_dir():
            raise ValueError(f"Sample root does not exist: {self.sample_root}")

        expected_names = {
            FRAME_TEMPLATE.format(frame=frame) for frame in range(self.expected_frames)
        }
        actual_names = {path.name for path in self.sample_root.glob("*.png")}
        unexpected_names = sorted(actual_names - expected_names)
        if unexpected_names:
            raise ValueError(
                f"{self.sample_root}: unexpected PNG frames: {unexpected_names}"
            )

        for frame in range(self.expected_frames):
            path = self.sample_root / FRAME_TEMPLATE.format(frame=frame)
            if not path.is_file():
                raise ValueError(f"{self.sample_root}: missing required frame {path.name}")
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
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index != 0:
            raise IndexError(index)
        frames = [
            self._load_rgb(self.sample_root / FRAME_TEMPLATE.format(frame=frame))
            for frame in range(self.expected_frames)
        ]
        return {
            "video": torch.stack(frames),
            "prompt": self.prompt,
            "sample_id": self.sample_root.parent.name,
        }

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).mul_(2.0).sub_(1.0)
