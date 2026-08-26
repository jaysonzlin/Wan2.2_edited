"""Native 480-square SimGen samples for history-conditioned joint training."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.simgen_utonia_features import SimGenUtoniaCache


def simgen_joint_collate(samples: list[dict[str, object]]) -> dict[str, object]:
    """Collate the variable-object batch-size-one training route."""
    if len(samples) != 1:
        raise ValueError("simgen_joint_collate supports train_batch_size=1 only")
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in samples[0].items()
    }


class SimGenJointDataset(Dataset):
    """Read specified SimGen samples without altering their native RGB geometry."""

    def __init__(
        self,
        dataset_root: str | Path,
        sample_ids: list[int],
        *,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (480, 480),
        expected_points: int = 2048,
        utonia_cache_root: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self.expected_points = expected_points
        self.utonia_cache = (
            None if utonia_cache_root is None else SimGenUtoniaCache(utonia_cache_root)
        )
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")
        self.sample_dirs = [self.dataset_root / f"sample_{sample_id}" for sample_id in sample_ids]
        missing = [path for path in self.sample_dirs if not path.is_dir()]
        if missing:
            raise ValueError(f"Missing configured SimGen sample: {missing[0]}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_dir = self.sample_dirs[index]
        metadata = self._load_metadata(sample_dir / "metadata.json")
        objects = self._metadata_objects(metadata, sample_dir / "metadata.json")
        object_dirs = self._object_dirs(sample_dir / "objects", objects)
        frames = [
            self._load_frame(sample_dir / "view_0" / f"{frame:08d}.png")
            for frame in range(self.expected_frames)
        ]
        points = [self._load_points(path / "pc.hdf5") for path in object_dirs]
        item = {
            "video": torch.stack(frames),
            "point_clouds": torch.stack(points),
            "object_ids": [path.name for path in object_dirs],
            "object_names": [objects[path.name] for path in object_dirs],
            "metadata": metadata,
            "sample_id": sample_dir.name,
        }
        if self.utonia_cache is not None:
            item["utonia_features"] = torch.stack(
                [
                    self.utonia_cache.features_for(name, point[0, 0])
                    for name, point in zip(item["object_names"], points, strict=True)
                ]
            )
        return item

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

    @staticmethod
    def _load_metadata(path: Path) -> dict:
        try:
            with path.open() as source:
                metadata = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid JSON metadata") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: metadata.json must contain a JSON object")
        return metadata

    @staticmethod
    def _metadata_objects(metadata: dict, path: Path) -> dict[str, str]:
        entries = metadata.get("instances", metadata.get("objects"))
        if not isinstance(entries, list):
            raise ValueError(f"{path}: metadata must contain an instances list")
        names: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("name"), str):
                raise ValueError(f"{path}: every metadata instance needs string id and name")
            if entry["id"] in names:
                raise ValueError(f"{path}: duplicate metadata object id {entry['id']}")
            names[entry["id"]] = entry["name"]
        return names

    @staticmethod
    def _object_dirs(root: Path, names: dict[str, str]) -> list[Path]:
        paths = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        if not paths:
            raise ValueError(f"{root.parent}: no object directories found in objects")
        unknown = [path.name for path in paths if path.name not in names]
        if unknown:
            raise ValueError(f"{root}: object directory missing metadata mapping: {unknown[0]}")
        return paths

    def _load_points(self, path: Path) -> torch.Tensor:
        try:
            with h5py.File(path, "r") as source:
                points = np.asarray(source["point_cloud"][:], dtype=np.float32)
        except (KeyError, OSError) as error:
            raise ValueError(f"Unable to read point cloud from {path}") from error
        expected = (self.expected_frames, 1, self.expected_points, 3)
        if points.shape != expected:
            raise ValueError(f"{path}: point_cloud must have shape {expected}")
        if not np.isfinite(points).all():
            raise ValueError(f"{path}: point_cloud contains non-finite values")
        return torch.from_numpy(points)
