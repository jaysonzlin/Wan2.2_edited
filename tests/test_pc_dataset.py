import h5py
import numpy as np
import pytest
import torch

from training.pc_dataset import PCTrajectoryDataset
from training.utonia_features import prepare_utonia_feature_cache


def write_pc_sample(path, shape=(49, 1, 2048, 3), rgb=None):
    path.mkdir(parents=True)
    with h5py.File(path / "pc.hdf5", "w") as source:
        source.create_dataset("point_cloud", data=np.zeros(shape, dtype=np.float32))
        source.create_dataset(
            "initial_linear_velocity", data=np.zeros((1, 3), dtype=np.float32)
        )
        source.create_dataset(
            "initial_angular_velocity", data=np.zeros((1, 3), dtype=np.float32)
        )
        if rgb is not None:
            source.create_dataset("rgb", data=rgb)


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


def test_object_dataset_selects_requested_object_and_requires_rgb(tmp_path):
    rgb = np.full((2048, 3), 127, dtype=np.uint8)
    write_pc_sample(tmp_path / "sample_0" / "objects" / "000", rgb=rgb)

    sample = PCTrajectoryDataset(tmp_path, object_id="000")[0]

    assert sample["sample_id"] == "sample_0/objects/000"
    assert sample["points_src"].shape == (1, 2048, 3)


def test_object_dataset_rejects_missing_requested_object(tmp_path):
    write_pc_sample(tmp_path / "sample_0" / "objects" / "001", rgb=np.zeros((2048, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="objects/000: missing required file pc.hdf5"):
        PCTrajectoryDataset(tmp_path, object_id="000")


@pytest.mark.parametrize(
    "rgb",
    [
        None,
        np.zeros((2048, 3), dtype=np.float32),
        np.zeros((2048, 2), dtype=np.uint8),
    ],
)
def test_object_dataset_rejects_invalid_rgb(tmp_path, rgb):
    write_pc_sample(tmp_path / "sample_0" / "objects" / "000", rgb=rgb)

    with pytest.raises((KeyError, ValueError), match="rgb"):
        PCTrajectoryDataset(tmp_path, object_id="000")


def test_object_dataset_loads_prepared_utonia_features(tmp_path):
    source_path = tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5"
    write_pc_sample(source_path.parent, rgb=np.zeros((2048, 3), dtype=np.uint8))

    class FakeExtractor:
        checkpoint_fingerprint = "checkpoint"
        preprocess_version = "preprocess"

        def __call__(self, coordinates, rgb):
            return torch.ones((2048, 5))

    cache_root = tmp_path / "utonia_cache"
    prepare_utonia_feature_cache(
        {"sample_0/objects/000": source_path}, cache_root, FakeExtractor()
    )

    sample = PCTrajectoryDataset(
        tmp_path, object_id="000", utonia_cache_root=cache_root
    )[0]

    assert sample["utonia_features"].shape == (2048, 5)
    assert sample["utonia_features"].dtype == torch.float32
