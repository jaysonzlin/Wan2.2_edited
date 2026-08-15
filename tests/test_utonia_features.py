import types

import h5py
import numpy as np
import pytest
import torch
import wan

from training.utonia_features import (
    UtoniaFeatureExtractor,
    load_cached_utonia_features,
    prepare_utonia_color,
    prepare_utonia_feature_cache,
)


def write_object_cloud(path, *, coordinate_value=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = np.full((49, 1, 2048, 3), coordinate_value, dtype=np.float32)
    with h5py.File(path, "w") as source:
        source.create_dataset("point_cloud", data=point_cloud)
        source.create_dataset(
            "rgb", data=np.arange(2048 * 3, dtype=np.uint8).reshape(2048, 3)
        )


class FakeExtractor:
    checkpoint_fingerprint = "fake-checkpoint-v1"
    preprocess_version = "fake-preprocess-v1"

    def __init__(self):
        self.calls = 0

    def __call__(self, coordinates, rgb):
        self.calls += 1
        assert coordinates.shape == (2048, 3)
        assert rgb.dtype == np.uint8
        return torch.from_numpy(np.concatenate([coordinates, rgb[:, :1]], axis=1))


def test_extractor_uses_vendored_utonia(monkeypatch, tmp_path):
    checkpoint = tmp_path / "_utonia_checkpoint" / "utonia.pth"

    class FakeModel:
        def cuda(self):
            return self

        def eval(self):
            return self

    def load(*_args, **kwargs):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")
        assert kwargs["repo_id"] == "Pointcept/Utonia"
        return FakeModel()

    fake_utonia = types.SimpleNamespace(
        model=types.SimpleNamespace(load=load),
        transform=types.SimpleNamespace(default=lambda **_kwargs: object()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan, "utonia", fake_utonia, raising=False)

    extractor = UtoniaFeatureExtractor(tmp_path, training_seed=42)

    assert extractor.model.__class__ is FakeModel
    assert extractor.training_seed == 42
    assert extractor.preprocess_version.endswith("training-seed-42")


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        (
            np.array([[0, 127, 255]], dtype=np.uint8),
            np.array([[0.0, 127.0, 255.0]], dtype=np.float32),
        ),
        (
            np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
            np.array([[0.0, 127.5, 255.0]], dtype=np.float32),
        ),
    ],
)
def test_prepare_utonia_color_preserves_uint8_and_scales_unit_float_rgb(rgb, expected):
    prepared = prepare_utonia_color(rgb)

    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, expected)


@pytest.mark.parametrize(
    "rgb",
    [
        np.array([[0.0, 0.5, 1.1]], dtype=np.float32),
        np.array([[0.0, np.nan, 1.0]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 3), dtype=np.int16),
    ],
)
def test_prepare_utonia_color_rejects_invalid_stored_rgb(rgb):
    with pytest.raises(ValueError, match="rgb"):
        prepare_utonia_color(rgb)


def test_prepare_cache_reuses_matching_entry_and_preserves_point_order(tmp_path):
    path = tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5"
    write_object_cloud(path)
    extractor = FakeExtractor()
    sources = {"sample_0/objects/000": path}

    assert prepare_utonia_feature_cache(sources, tmp_path / "cache", extractor) == 4
    first = load_cached_utonia_features(tmp_path / "cache", "sample_0/objects/000")
    assert extractor.calls == 1
    torch.testing.assert_close(first[:, :3], torch.zeros(2048, 3))
    expected_color = np.arange(2048 * 3, dtype=np.uint8).reshape(2048, 3)[:, 0]
    torch.testing.assert_close(first[:, 3], torch.from_numpy(expected_color).float())

    assert prepare_utonia_feature_cache(sources, tmp_path / "cache", extractor) == 4
    assert extractor.calls == 1


def test_prepare_cache_rebuilds_after_source_or_extractor_changes(tmp_path):
    path = tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5"
    write_object_cloud(path)
    sources = {"sample_0/objects/000": path}
    extractor = FakeExtractor()

    prepare_utonia_feature_cache(sources, tmp_path / "cache", extractor)
    write_object_cloud(path, coordinate_value=2.0)
    prepare_utonia_feature_cache(sources, tmp_path / "cache", extractor)
    assert extractor.calls == 2

    extractor.checkpoint_fingerprint = "fake-checkpoint-v2"
    prepare_utonia_feature_cache(sources, tmp_path / "cache", extractor)
    assert extractor.calls == 3


def test_prepare_cache_rejects_invalid_extractor_features(tmp_path):
    path = tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5"
    write_object_cloud(path)

    class InvalidExtractor(FakeExtractor):
        def __call__(self, coordinates, rgb):
            return torch.zeros((2047, 4))

    with pytest.raises(ValueError, match=r"features must have shape \(2048, D\)"):
        prepare_utonia_feature_cache(
            {"sample_0/objects/000": path}, tmp_path / "cache", InvalidExtractor()
        )


def test_prepare_cache_rejects_inconsistent_feature_widths(tmp_path):
    first = tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5"
    second = tmp_path / "sample_1" / "objects" / "000" / "pc.hdf5"
    write_object_cloud(first)
    write_object_cloud(second)

    class VariableWidthExtractor(FakeExtractor):
        def __call__(self, coordinates, rgb):
            self.calls += 1
            return torch.zeros((2048, 3 + self.calls))

    with pytest.raises(ValueError, match="feature width"):
        prepare_utonia_feature_cache(
            {"sample_0/objects/000": first, "sample_1/objects/000": second},
            tmp_path / "cache",
            VariableWidthExtractor(),
        )
