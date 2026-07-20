import h5py
import numpy as np
import pytest

from training.pc_dataset import PCTrajectoryDataset


def write_pc_sample(path, shape=(49, 1, 2048, 3)):
    path.mkdir(parents=True)
    with h5py.File(path / "pc.hdf5", "w") as source:
        source.create_dataset("point_cloud", data=np.zeros(shape, dtype=np.float32))
        source.create_dataset(
            "initial_linear_velocity", data=np.zeros((1, 3), dtype=np.float32)
        )
        source.create_dataset(
            "initial_angular_velocity", data=np.zeros((1, 3), dtype=np.float32)
        )


def test_dataset_splits_a_valid_hdf5_clip(tmp_path):
    write_pc_sample(tmp_path / "sample_0")

    sample = PCTrajectoryDataset(tmp_path)[0]

    assert sample["points_src"].shape == (1, 2048, 3)
    assert sample["points_tgt"].shape == (48, 1, 2048, 3)
    assert sample["initial_angular_velocity"].shape == (1, 3)


def test_dataset_rejects_wrong_point_shape(tmp_path):
    write_pc_sample(tmp_path / "sample_0", shape=(49, 1, 8, 3))

    with pytest.raises(
        ValueError, match=r"point_cloud must have shape \(49, 1, 2048, 3\)"
    ):
        PCTrajectoryDataset(tmp_path)
