"""Strict HDF5 dataset for fixed-size point-cloud trajectory clips."""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class PCTrajectoryDataset(Dataset):
    """Load PhysCtrl-format ``sample_*/pc.hdf5`` point-cloud trajectories."""

    def __init__(self, dataset_root: str | Path, expected_frames: int = 49, expected_points: int = 2048):
        self.dataset_root = Path(dataset_root)
        self.expected_frames = expected_frames
        self.expected_points = expected_points
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")
        self.samples = sorted(path / "pc.hdf5" for path in self.dataset_root.glob("sample_*") if path.is_dir())
        if not self.samples:
            raise ValueError(f"No sample_* directories found in {self.dataset_root}")
        for path in self.samples:
            self._validate(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.samples[index]
        with h5py.File(path, "r") as source:
            point_cloud = torch.from_numpy(source["point_cloud"][:]).float()
            linear_velocity = torch.from_numpy(source["initial_linear_velocity"][:]).float()
            angular_velocity = torch.from_numpy(source["initial_angular_velocity"][:]).float()
        return {
            "points_src": point_cloud[0],
            "points_tgt": point_cloud[1:],
            "initial_linear_velocity": linear_velocity,
            "initial_angular_velocity": angular_velocity,
            "sample_id": path.parent.name,
        }

    def _validate(self, path: Path) -> None:
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
