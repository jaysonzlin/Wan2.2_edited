"""Dataset support for frame-aligned RGB-D point-view conditioning."""

from pathlib import Path

import h5py
import numpy as np
import torch

from training.pc_dataset import PCTrajectoryDataset
from training.utonia_features import validate_utonia_rgb


POINT_VIEW_FRAMES = 49
POINT_VIEW_POINTS = 2048


def read_point_view(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one variable-size point view after validating its stored schema."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as source:
            xyz = np.asarray(source["xyz"])
            rgb = np.asarray(source["rgb"])
    except (KeyError, OSError) as error:
        raise ValueError(f"{path}: point view must contain xyz and rgb datasets") from error
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError(f"{path}: xyz must have shape (N, 3)")
    if not np.issubdtype(xyz.dtype, np.floating):
        raise ValueError(f"{path}: xyz must have a floating-point dtype")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: xyz must be finite")
    if xyz.shape[0] > POINT_VIEW_POINTS:
        raise ValueError(f"{path}: point view must contain at most 2048 points")
    try:
        validate_utonia_rgb(rgb, point_count=xyz.shape[0])
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return np.asarray(xyz, dtype=np.float32), rgb


def point_view_paths(pc_path: str | Path, *, object_id: str) -> tuple[Path, ...]:
    """Return the strict ordered point-view file set for an object PC source."""
    pc_path = Path(pc_path)
    sample_dir = pc_path.parent.parent.parent
    paths = tuple(sample_dir / "point_views" / f"{frame:04d}.h5" for frame in range(POINT_VIEW_FRAMES))
    for path in paths:
        if not path.is_file():
            raise ValueError(f"{path}: missing required point-view frame")
    return paths


def pack_point_views(paths: tuple[Path, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ordered variable-size views into fixed tensors and a validity mask."""
    if len(paths) != POINT_VIEW_FRAMES:
        raise ValueError("point views must contain exactly 49 frames")
    views = torch.zeros((POINT_VIEW_FRAMES, POINT_VIEW_POINTS, 3), dtype=torch.float32)
    mask = torch.zeros((POINT_VIEW_FRAMES, POINT_VIEW_POINTS), dtype=torch.bool)
    for frame, path in enumerate(paths):
        xyz, _ = read_point_view(path)
        count = xyz.shape[0]
        if count:
            views[frame, :count] = torch.from_numpy(xyz)
            mask[frame, :count] = True
    return views, mask


class PVCTrajectoryDataset(PCTrajectoryDataset):
    """History trajectory clips with all per-frame view conditions."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        object_id: str,
        utonia_cache_root: str | Path | None = None,
        point_view_utonia_cache_root: str | Path | None = None,
    ):
        super().__init__(
            dataset_root,
            history_frames=4,
            object_id=object_id,
            utonia_cache_root=utonia_cache_root,
        )
        self.point_view_utonia_cache_root = (
            Path(point_view_utonia_cache_root)
            if point_view_utonia_cache_root is not None
            else None
        )
        self.point_view_source_paths = {
            self._sample_id(path): point_view_paths(path, object_id=object_id)
            for path in self.samples
        }
        for paths in self.point_view_source_paths.values():
            for path in paths:
                read_point_view(path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        paths = self.point_view_source_paths[sample["sample_id"]]
        views, mask = pack_point_views(paths)
        sample["point_views"] = views
        sample["point_view_mask"] = mask
        return sample
