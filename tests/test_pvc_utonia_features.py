import h5py
import numpy as np
import torch

from training.pvc_utonia_features import (
    load_cached_point_view_utonia_features,
    prepare_point_view_utonia_feature_cache,
)


class RecordingExtractor:
    checkpoint_fingerprint = "checkpoint-v1"
    preprocess_version = "preprocess-v1"

    def __init__(self):
        self.row_counts = []

    def __call__(self, xyz, rgb):
        self.row_counts.append(len(xyz))
        return torch.from_numpy(np.concatenate((xyz, rgb[:, :1]), axis=1))


def write_views(root, counts):
    paths = []
    for frame, count in enumerate(counts):
        path = root / "point_views" / f"{frame:04d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as output:
            output.create_dataset("xyz", data=np.full((count, 3), frame, dtype=np.float32))
            output.create_dataset("rgb", data=np.full((count, 3), 127, dtype=np.uint8))
        paths.append(path)
    return tuple(paths)


def test_point_view_cache_encodes_real_rows_then_pads_and_reuses(tmp_path):
    sources = {"sample_0/objects/000": write_views(tmp_path, [2] + [0] * 48)}
    extractor = RecordingExtractor()

    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=4)
    features, mask = load_cached_point_view_utonia_features(
        tmp_path / "cache", "sample_0/objects/000", feature_dim=4
    )

    assert features.shape == (49, 2048, 4)
    assert mask.shape == (49, 2048)
    assert mask.sum().item() == 2
    assert extractor.row_counts == [2]
    torch.testing.assert_close(features[0, :2, :3], torch.zeros((2, 3)))
    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=4)
    assert extractor.row_counts == [2]


def test_point_view_cache_rebuilds_when_any_view_source_changes(tmp_path):
    paths = write_views(tmp_path, [1] * 49)
    sources = {"sample_0/objects/000": paths}
    extractor = RecordingExtractor()

    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=4)
    with h5py.File(paths[12], "w") as output:
        output.create_dataset("xyz", data=np.full((2, 3), 12, dtype=np.float32))
        output.create_dataset("rgb", data=np.full((2, 3), 127, dtype=np.uint8))
    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=4)

    assert len(extractor.row_counts) == 98


def test_point_view_cache_allows_all_empty_views_with_known_width(tmp_path):
    sources = {"sample_0/objects/000": write_views(tmp_path, [0] * 49)}
    extractor = RecordingExtractor()

    prepare_point_view_utonia_feature_cache(sources, tmp_path / "cache", extractor, feature_dim=4)
    features, mask = load_cached_point_view_utonia_features(
        tmp_path / "cache", "sample_0/objects/000", feature_dim=4
    )

    assert not mask.any()
    assert not features.any()
    assert extractor.row_counts == []
