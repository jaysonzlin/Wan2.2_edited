import json

import h5py
import numpy as np
import torch
from PIL import Image

from training.simgen_joint_dataset import SimGenJointDataset


def _write_sample(root, sample_id="sample_0"):
    sample = root / sample_id
    view_root = sample / "view_0"
    view_root.mkdir(parents=True)
    for frame in range(49):
        Image.new("RGB", (480, 480), color=(frame, 2, 3)).save(
            view_root / f"{frame:08d}.png"
        )
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"id": "001", "name": "ball", "instance_id": "ball_0"},
                    {"id": "000", "name": "panda", "instance_id": "panda_0"},
                ]
            }
        )
    )
    for ordinal, value in (("001", 1.0), ("000", 0.0)):
        object_dir = sample / "objects" / ordinal
        object_dir.mkdir(parents=True)
        with h5py.File(object_dir / "pc.hdf5", "w") as output:
            output.create_dataset(
                "point_cloud",
                data=np.full((49, 1, 2, 3), value, dtype=np.float32),
            )
            output.create_dataset("rgb", data=np.full((2, 3), value, dtype=np.float32))
    return sample


def test_simgen_dataset_loads_native_rgb_and_metadata_object_order(tmp_path):
    _write_sample(tmp_path)

    item = SimGenJointDataset(tmp_path, sample_ids=[0], expected_points=2)[0]

    assert item["sample_id"] == "sample_0"
    assert item["video"].shape == (49, 3, 480, 480)
    assert item["video"][0, :, 0, 0].tolist() == [-1.0, -0.9843137264251709, -0.9764705896377563]
    assert item["object_ids"] == ["000", "001"]
    assert item["object_names"] == ["panda", "ball"]
    assert item["point_clouds"].shape == (2, 49, 1, 2, 3)
    assert item["point_clouds"][0].eq(0).all()
    assert item["point_clouds"][1].eq(1).all()


def test_simgen_dataset_attaches_class_cache_features_in_object_order(tmp_path):
    _write_sample(tmp_path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    for name, feature in (("panda", 3.0), ("ball", 7.0)):
        torch.save(
            {
                    "metadata": {
                        "class_name": name, "point_count": 2, "feature_dim": 1,
                        "record_version": 1, "normalization_version": "centroid-rms-v1",
                        "source_fingerprint": "source-v1", "checkpoint_fingerprint": "weights-v1",
                        "preprocess_version": "preprocess-v1",
                    },
                "reference_points": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "features": torch.full((2, 1), feature),
            },
            cache_root / f"{name}.pt",
        )

    item = SimGenJointDataset(tmp_path, sample_ids=[0], expected_points=2, utonia_cache_root=cache_root)[0]

    assert item["utonia_features"].shape == (2, 2, 1)
    assert item["utonia_features"][0].eq(3).all()
    assert item["utonia_features"][1].eq(7).all()
