import h5py
import numpy as np
import pytest
import torch

from training.simgen_utonia_features import (
    SimGenUtoniaCache,
    prepare_simgen_utonia_cache,
    transfer_dense_features,
)


def test_transfer_dense_features_uses_centroid_and_rms_normalized_nearest_neighbors():
    reference_points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    reference_features = torch.tensor([[10.0, 1.0], [20.0, 2.0]])
    target_points = torch.tensor([[11.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    transferred = transfer_dense_features(
        reference_points, reference_features, target_points
    )

    assert torch.equal(transferred, torch.tensor([[20.0, 2.0], [10.0, 1.0]]))


def test_class_cache_loads_dense_reference_record_and_transfers_features(tmp_path):
    torch.save(
        {
            "metadata": {
                "class_name": "panda", "feature_dim": 2, "point_count": 2,
                "record_version": 1, "normalization_version": "centroid-rms-v1",
                "source_fingerprint": "source-v1", "checkpoint_fingerprint": "weights-v1",
                "preprocess_version": "preprocess-v1",
            },
            "reference_points": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "features": torch.tensor([[10.0, 1.0], [20.0, 2.0]]),
        },
        tmp_path / "panda.pt",
    )

    features = SimGenUtoniaCache(tmp_path).features_for(
        "panda", torch.tensor([[11.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    )

    assert torch.equal(features, torch.tensor([[20.0, 2.0], [10.0, 1.0]]))


def test_prepare_cache_writes_class_keyed_canonical_record(tmp_path):
    source = tmp_path / "pc.hdf5"
    with h5py.File(source, "w") as output:
        output.create_dataset("point_cloud", data=np.zeros((49, 1, 2, 3), dtype=np.float32))
        output.create_dataset("rgb", data=np.ones((2, 3), dtype=np.float32))

    class Extractor:
        checkpoint_fingerprint = "weights-v1"
        preprocess_version = "preprocess-v1"

        def __call__(self, xyz, rgb):
            assert xyz.shape == (2, 3) and rgb.shape == (2, 3)
            return torch.ones(2, 3)

    assert prepare_simgen_utonia_cache({"panda": source}, tmp_path / "cache", Extractor()) == 3
    record = torch.load(tmp_path / "cache" / "panda.pt", weights_only=True)
    assert record["metadata"]["class_name"] == "panda"
    assert record["metadata"]["checkpoint_fingerprint"] == "weights-v1"


def test_prepare_cache_reuses_only_matching_fingerprinted_records(tmp_path):
    source = tmp_path / "pc.hdf5"
    with h5py.File(source, "w") as output:
        output.create_dataset("point_cloud", data=np.zeros((49, 1, 2, 3), dtype=np.float32))
        output.create_dataset("rgb", data=np.ones((2, 3), dtype=np.float32))

    class Extractor:
        checkpoint_fingerprint = "weights-v1"
        preprocess_version = "preprocess-v1"

        def __init__(self):
            self.calls = 0

        def __call__(self, xyz, rgb):
            self.calls += 1
            return torch.ones(2, 3)

    extractor = Extractor()
    assert prepare_simgen_utonia_cache({"panda": source}, tmp_path / "cache", extractor) == 3
    assert prepare_simgen_utonia_cache({"panda": source}, tmp_path / "cache", extractor) == 3
    assert extractor.calls == 1


def test_cache_rejects_legacy_and_unknown_class_records(tmp_path):
    torch.save(
        {
            "metadata": {"class_name": "panda", "feature_dim": 2, "point_count": 2},
            "reference_points": torch.zeros(2, 3), "features": torch.zeros(2, 2),
        },
        tmp_path / "panda.pt",
    )

    with pytest.raises(ValueError, match="record version"):
        SimGenUtoniaCache(tmp_path).features_for("panda", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="Unknown canonical Utonia class"):
        SimGenUtoniaCache(tmp_path).features_for("mug", torch.zeros(2, 3))
