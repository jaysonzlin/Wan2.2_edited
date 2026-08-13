"""Strict HDF5 dataset for fixed-size point-cloud trajectory clips."""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from training.utonia_features import load_cached_utonia_features


class PCTrajectoryDataset(Dataset):
    """Load PhysCtrl-format ``sample_*/pc.hdf5`` point-cloud trajectories."""

    def __init__(
        self,
        dataset_root: str | Path,
        expected_frames: int = 49,
        expected_points: int = 2048,
        *,
        object_id: str | None = None,
        utonia_cache_root: str | Path | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.expected_frames = expected_frames
        self.expected_points = expected_points
        self.object_id = object_id
        self.utonia_cache_root = (
            Path(utonia_cache_root) if utonia_cache_root is not None else None
        )
        if self.utonia_cache_root is not None and object_id is None:
            raise ValueError("utonia_cache_root requires object_id")
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")
        sample_directories = sorted(
            path for path in self.dataset_root.glob("sample_*") if path.is_dir()
        )
        if object_id is None:
            self.samples = [path / "pc.hdf5" for path in sample_directories]
        else:
            self.samples = [
                path / "objects" / object_id / "pc.hdf5"
                for path in sample_directories
            ]
        if not self.samples:
            raise ValueError(f"No sample_* directories found in {self.dataset_root}")
        for path in self.samples:
            self._validate(path, require_rgb=object_id is not None)
        self.source_paths = {self._sample_id(path): path for path in self.samples}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.samples[index]
        with h5py.File(path, "r") as source:
            point_cloud = torch.from_numpy(source["point_cloud"][:]).float()
            linear_velocity = torch.from_numpy(source["initial_linear_velocity"][:]).float()
            angular_velocity = torch.from_numpy(source["initial_angular_velocity"][:]).float()
        sample = {
            "points_src": point_cloud[0],
            "points_tgt": point_cloud[1:],
            "initial_linear_velocity": linear_velocity,
            "initial_angular_velocity": angular_velocity,
            "sample_id": self._sample_id(path),
        }
        if self.utonia_cache_root is not None:
            sample["utonia_features"] = load_cached_utonia_features(
                self.utonia_cache_root, sample["sample_id"]
            )
        return sample

    def _sample_id(self, path: Path) -> str:
        if self.object_id is None:
            return path.parent.name
        return f"{path.parent.parent.parent.name}/objects/{self.object_id}"

    def _validate(self, path: Path, *, require_rgb: bool) -> None:
        if not path.is_file():
            raise ValueError(f"{path.parent}: missing required file pc.hdf5")
        with h5py.File(path, "r") as source:
            required = ("point_cloud", "initial_linear_velocity", "initial_angular_velocity")
            missing = [key for key in required if key not in source]
            if missing:
                raise KeyError(f"{path}: missing required datasets: {', '.join(missing)}")
            expected_cloud = (self.expected_frames, 1, self.expected_points, 3)
            if source["point_cloud"].shape != expected_cloud:
                raise ValueError(f"{path}: point_cloud must have shape {expected_cloud}")
            for key in ("initial_linear_velocity", "initial_angular_velocity"):
                if source[key].shape != (1, 3):
                    raise ValueError(f"{path}: {key} must have shape (1, 3)")
            if require_rgb:
                if "rgb" not in source:
                    raise KeyError(f"{path}: missing required dataset rgb")
                rgb = source["rgb"]
                expected_rgb = (self.expected_points, 3)
                if rgb.shape != expected_rgb:
                    raise ValueError(f"{path}: rgb must have shape {expected_rgb}")
                if rgb.dtype != "uint8":
                    raise ValueError(f"{path}: rgb must have dtype uint8")
                if rgb[:].min() < 0 or rgb[:].max() > 255:
                    raise ValueError(f"{path}: rgb values must be in [0, 255]")
