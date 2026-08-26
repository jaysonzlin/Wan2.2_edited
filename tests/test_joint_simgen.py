import json
from pathlib import Path

import joint_simgen


def test_prepare_cache_uses_sample_zero_panda_ball_can_sources(monkeypatch, tmp_path):
    metadata_path = tmp_path / "sample_0" / "metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"id": "001", "name": "panda"},
                    {"id": "000", "name": "ball"},
                    {"id": "002", "name": "can"},
                ]
            }
        )
    )
    config = {
        "data": {"dataset_root": str(tmp_path), "utonia_cache_root": str(tmp_path / "cache")},
        "training": {"seed": 0},
    }
    calls = []
    monkeypatch.setattr(joint_simgen, "UtoniaFeatureExtractor", lambda root, seed: object())
    monkeypatch.setattr(
        joint_simgen, "prepare_simgen_utonia_cache",
        lambda sources, root, extractor: calls.append((sources, Path(root))) or 8,
    )

    assert joint_simgen.prepare_cache(config) == 8
    assert calls[0][0] == {
        "panda": tmp_path / "sample_0" / "objects" / "001" / "pc.hdf5",
        "ball": tmp_path / "sample_0" / "objects" / "000" / "pc.hdf5",
        "can": tmp_path / "sample_0" / "objects" / "002" / "pc.hdf5",
    }


def test_build_datasets_uses_fixed_configured_train_and_validation_ranges(monkeypatch):
    calls = []

    class Dataset:
        def __init__(self, root, sample_ids, **kwargs):
            calls.append((root, sample_ids, kwargs))

    monkeypatch.setattr(joint_simgen, "SimGenJointDataset", Dataset)
    config = {
        "data": {
            "dataset_root": "simgen", "train_start": 0, "train_end": 489,
            "validation_start": 490, "validation_end": 499,
            "num_points": 2048, "utonia_cache_root": "cache",
        }
    }

    joint_simgen.build_datasets(config)

    assert calls[0][1] == list(range(490))
    assert calls[1][1] == list(range(490, 500))
    assert calls[0][2]["utonia_cache_root"] == "cache"


def test_main_dispatches_training_when_cache_preparation_is_not_requested(monkeypatch):
    monkeypatch.setattr(joint_simgen, "load_simgen_joint_config", lambda path, overrides: {"mode": "train"})
    calls = []
    monkeypatch.setattr(joint_simgen, "run_training", lambda config: calls.append(config))
    monkeypatch.setattr("sys.argv", ["joint_simgen.py", "--config", "config.yaml"])

    joint_simgen.main()

    assert calls == [{"mode": "train"}]


def test_readme_documents_simgen_cache_preparation_and_training():
    readme = Path("README.md").read_text()

    assert "joint_simgen.py --prepare-utonia-cache" in readme
    assert "joint_simgen_480.yaml" in readme
