import h5py
import numpy as np
import pytest
import torch

from training.pvc_dataset import PVCTrajectoryDataset


def write_pc(path):
    path.mkdir(parents=True)
    with h5py.File(path / "pc.hdf5", "w") as source:
        source.create_dataset("point_cloud", data=np.zeros((49, 1, 2048, 3), dtype=np.float32))
        source.create_dataset("initial_linear_velocity", data=np.zeros((1, 3), dtype=np.float32))
        source.create_dataset("initial_angular_velocity", data=np.zeros((1, 3), dtype=np.float32))
        source.create_dataset("rgb", data=np.zeros((2048, 3), dtype=np.uint8))


def write_view(path, count, *, xyz=None, rgb=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "xyz",
            data=np.arange(count * 3, dtype=np.float32).reshape(count, 3)
            if xyz is None else xyz,
        )
        output.create_dataset(
            "rgb",
            data=np.full((count, 3), 127, dtype=np.uint8) if rgb is None else rgb,
        )


def write_sample(root, counts=None):
    sample = root / "sample_0"
    write_pc(sample / "objects" / "000")
    for frame, count in enumerate(counts or [1] * 49):
        write_view(sample / "point_views" / f"{frame:04d}.h5", count)
    return sample


def test_pvc_dataset_pads_each_view_and_preserves_validity(tmp_path):
    write_sample(tmp_path, [2, 0] + [1] * 47)

    sample = PVCTrajectoryDataset(tmp_path, object_id="000")[0]

    assert sample["point_views"].shape == (49, 2048, 3)
    assert sample["point_view_mask"].shape == (49, 2048)
    assert sample["point_view_mask"].dtype == torch.bool
    assert sample["point_view_mask"].sum(dim=1).tolist()[:3] == [2, 0, 1]
    torch.testing.assert_close(
        sample["point_views"][0, :2],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32),
    )
    assert not sample["point_view_mask"][0, 2]


def test_pvc_dataset_exposes_ordered_point_view_sources(tmp_path):
    sample_dir = write_sample(tmp_path)

    dataset = PVCTrajectoryDataset(tmp_path, object_id="000")

    paths = dataset.point_view_source_paths["sample_0/objects/000"]
    assert len(paths) == 49
    assert paths[0] == sample_dir / "point_views" / "0000.h5"
    assert paths[-1] == sample_dir / "point_views" / "0048.h5"


def test_pvc_dataset_rejects_missing_point_view_frame(tmp_path):
    sample = write_sample(tmp_path)
    (sample / "point_views" / "0007.h5").unlink()

    with pytest.raises(ValueError, match="0007.h5"):
        PVCTrajectoryDataset(tmp_path, object_id="000")


@pytest.mark.parametrize(
    ("xyz", "rgb", "message"),
    [
        (np.full((1, 3), np.nan, dtype=np.float32), None, "xyz must be finite"),
        (np.zeros((2049, 3), dtype=np.float32), np.zeros((2049, 3), dtype=np.uint8), "at most 2048"),
        (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.uint8), "rgb"),
    ],
)
def test_pvc_dataset_rejects_invalid_view_schema(tmp_path, xyz, rgb, message):
    sample = write_sample(tmp_path)
    write_view(sample / "point_views" / "0000.h5", len(xyz), xyz=xyz, rgb=rgb)

    with pytest.raises(ValueError, match=message):
        PVCTrajectoryDataset(tmp_path, object_id="000")


def test_pvc_dataset_does_not_require_depth_h5(tmp_path):
    write_sample(tmp_path)

    assert len(PVCTrajectoryDataset(tmp_path, object_id="000")) == 1
