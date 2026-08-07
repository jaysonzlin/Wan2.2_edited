"""Dataset for jointly training one video with its variable number of trajectories."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.overfit_dataset import FRAME_TEMPLATE


def joint_collate(samples: list[dict[str, object]]) -> dict[str, object]:
    """Collate the supported batch-size-one route without imposing a metadata schema."""
    if len(samples) != 1:
        raise ValueError("joint_collate supports train_batch_size=1 only")
    sample = samples[0]
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


class JointWanPhysCtrlDataset(Dataset):
    """Load complete RGBA clips and lexical-sorted object PC trajectories.

    Samples retain their variable object dimension. The initial joint trainer uses a batch size
    of one, so no object padding or object attention mask is introduced here.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        expected_frames: int = 49,
        expected_size: tuple[int, int] = (1280, 704),
        expected_points: int = 2048,
        load_deformation_fields: bool = False,
        expected_deform_neighbors: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self.expected_points = expected_points
        self.load_deformation_fields = load_deformation_fields
        self.expected_deform_neighbors = expected_deform_neighbors
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
        self._validate_video(sample_dir)
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"{sample_dir}: missing required file metadata.json")
        try:
            with metadata_path.open() as handle:
                metadata = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{metadata_path}: invalid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError(
                f"{metadata_path}: metadata.json must contain a JSON object"
            )

        object_root = sample_dir / "objects"
        object_dirs = (
            sorted(path for path in object_root.iterdir() if path.is_dir())
            if object_root.is_dir()
            else []
        )
        if not object_dirs:
            raise ValueError(f"{sample_dir}: no object directories found in objects")
        transforms = []
        for object_dir in object_dirs:
            self._validate_trajectory(object_dir / "pc.hdf5")
            if self.load_deformation_fields:
                transforms.append(
                    self._validate_deformation_contract(object_dir / "pc.hdf5")
                )
        if transforms and any(
            not np.array_equal(transform[0], transforms[0][0])
            or not np.array_equal(transform[1], transforms[0][1])
            or transform[2:] != transforms[0][2:]
            for transform in transforms[1:]
        ):
            raise ValueError(
                f"{sample_dir}: deformation grid metadata must agree across objects"
            )

    def _validate_video(self, sample_dir: Path) -> None:
        expected_names = {
            FRAME_TEMPLATE.format(frame=frame) for frame in range(self.expected_frames)
        }
        for frame in range(self.expected_frames):
            path = sample_dir / FRAME_TEMPLATE.format(frame=frame)
            if not path.is_file():
                raise ValueError(f"{sample_dir}: missing required frame {path.name}")
            with Image.open(path) as image:
                if image.mode != "RGBA":
                    raise ValueError(
                        f"{path}: expected RGBA PNG, got mode {image.mode}"
                    )
                if image.size != self.expected_size:
                    raise ValueError(
                        f"{path}: expected {self.expected_size[0]}x{self.expected_size[1]}, "
                        f"got {image.size[0]}x{image.size[1]}"
                    )
        unexpected = sorted(
            path.name
            for path in sample_dir.glob("rgba_*.png")
            if path.name not in expected_names
        )
        if unexpected:
            raise ValueError(f"{sample_dir}: unexpected RGBA frames: {unexpected}")

    def _validate_trajectory(self, path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"{path.parent}: missing required file pc.hdf5")
        with h5py.File(path, "r") as source:
            required = (
                "point_cloud",
                "initial_linear_velocity",
                "initial_angular_velocity",
            )
            missing = [key for key in required if key not in source]
            if missing:
                raise KeyError(
                    f"{path}: missing required datasets: {', '.join(missing)}"
                )
            expected_cloud = (self.expected_frames, 1, self.expected_points, 3)
            if source["point_cloud"].shape != expected_cloud:
                raise ValueError(
                    f"{path}: point_cloud must have shape {expected_cloud}"
                )
            for key in required[1:]:
                if source[key].shape != (1, 3):
                    raise ValueError(f"{path}: {key} must have shape (1, 3)")

    def _validate_deformation_contract(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray, float, int, float, int]:
        """Validate optional, precomputed approximate-PhysCtrl arrays for one object."""
        with h5py.File(path, "r") as source:
            required_shapes = {
                "deform_F": (self.expected_frames, 1, self.expected_points, 3, 3),
                "deform_C": (self.expected_frames, 1, self.expected_points, 3, 3),
                "deform_volume": (1, self.expected_points),
                "deform_baseline": (
                    self.expected_frames - 2,
                    1,
                    self.expected_points,
                    3,
                    3,
                ),
                "deform_grid_origin": (3,),
                "deform_grid_scale": (1,),
            }
            missing = [key for key in required_shapes if key not in source]
            if missing:
                raise KeyError(
                    f"{path}: missing required deformation datasets: {', '.join(missing)}"
                )
            for key, shape in required_shapes.items():
                dataset = source[key]
                if dataset.shape != shape:
                    raise ValueError(f"{path}: {key} must have shape {shape}")
                if dataset.dtype != np.float32:
                    raise ValueError(f"{path}: {key} must use float32")
                if not np.isfinite(dataset[:]).all():
                    raise ValueError(f"{path}: {key} contains non-finite values")
            attributes = (
                "deform_dt",
                "deform_grid_size",
                "deform_grid_lim",
                "deform_neighbors",
            )
            missing_attributes = [key for key in attributes if key not in source.attrs]
            if missing_attributes:
                raise KeyError(
                    f"{path}: missing required deformation attributes: {', '.join(missing_attributes)}"
                )
            dt = float(source.attrs["deform_dt"])
            grid_size = int(source.attrs["deform_grid_size"])
            grid_lim = float(source.attrs["deform_grid_lim"])
            neighbors = int(source.attrs["deform_neighbors"])
            if (
                not np.isfinite([dt, grid_lim]).all()
                or dt <= 0
                or grid_size < 5
                or grid_lim <= 0
            ):
                raise ValueError(f"{path}: invalid deformation grid metadata")
            if not 1 <= neighbors < self.expected_points:
                raise ValueError(
                    f"{path}: deform_neighbors must be in [1, expected_points)"
                )
            if (
                self.expected_deform_neighbors is not None
                and neighbors != self.expected_deform_neighbors
            ):
                raise ValueError(
                    f"{path}: deform_neighbors={neighbors} does not match the configured "
                    f"{self.expected_deform_neighbors}"
                )
            return (
                source["deform_grid_origin"][:],
                source["deform_grid_scale"][:],
                dt,
                grid_size,
                grid_lim,
                neighbors,
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_dir = self.sample_dirs[index]
        object_dirs = sorted(
            path for path in (sample_dir / "objects").iterdir() if path.is_dir()
        )
        points, linear_velocities, angular_velocities = [], [], []
        deform_fields: dict[str, list[torch.Tensor]] = {
            "deform_F": [],
            "deform_C": [],
            "deform_volume": [],
            "deform_baseline": [],
            "deform_grid_origin": [],
            "deform_grid_scale": [],
        }
        for object_dir in object_dirs:
            with h5py.File(object_dir / "pc.hdf5", "r") as source:
                points.append(torch.from_numpy(source["point_cloud"][:]).float())
                linear_velocities.append(
                    torch.from_numpy(source["initial_linear_velocity"][:]).float()
                )
                angular_velocities.append(
                    torch.from_numpy(source["initial_angular_velocity"][:]).float()
                )
                if self.load_deformation_fields:
                    for key, values in deform_fields.items():
                        values.append(torch.from_numpy(source[key][:]).float())
        with (sample_dir / "metadata.json").open() as handle:
            metadata = json.load(handle)
        frames = [
            self._load_frame(sample_dir / FRAME_TEMPLATE.format(frame=frame))
            for frame in range(self.expected_frames)
        ]
        item = {
            "video": torch.stack(frames),
            "point_clouds": torch.stack(points),
            "initial_linear_velocities": torch.stack(linear_velocities),
            "initial_angular_velocities": torch.stack(angular_velocities),
            "object_ids": [path.name for path in object_dirs],
            "metadata": metadata,
            "sample_id": sample_dir.name,
        }
        if self.load_deformation_fields:
            item.update(
                {key: torch.stack(values) for key, values in deform_fields.items()}
            )
        return item

    @staticmethod
    def _load_frame(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).mul_(2.0).sub_(1.0)
