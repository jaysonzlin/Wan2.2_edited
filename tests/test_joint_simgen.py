import json
from pathlib import Path

import joint_simgen
import pytest
import torch
import yaml


class SumReducer:
    def __init__(self, reduced_values):
        self.reduced_values = iter(reduced_values)

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return next(self.reduced_values)


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


def test_main_rejects_distributed_cache_preparation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(joint_simgen, "load_simgen_joint_config", lambda *_: {"data": {}})
    monkeypatch.setattr(
        "sys.argv", ["joint_simgen.py", "--config", "config.yaml", "--prepare-utonia-cache"]
    )

    with pytest.raises(ValueError, match="single GPU"):
        joint_simgen.main()


def test_shared_generator_restarts_from_configured_seed():
    first = joint_simgen._shared_generator("cpu", 42)
    second = joint_simgen._shared_generator("cpu", 42)

    assert torch.equal(torch.rand(4, generator=first), torch.rand(4, generator=second))


def test_reduced_mean_uses_global_loss_sum_and_example_count():
    accelerator = SumReducer([torch.tensor(5.0), torch.tensor(2)])

    assert joint_simgen._reduced_mean(accelerator, torch.tensor(5.0), 2).item() == 2.5


def test_validation_metrics_include_mean_pc_loss():
    metrics = joint_simgen._validation_metrics(
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(0.5), torch.tensor(1.5)],
    )

    assert metrics == {"validation/loss": 3.0, "validation/pc_loss_sum": 1.0}


def test_validation_loss_components_select_total_and_pc_loss_sum():
    total_loss, pc_loss_sum = joint_simgen._validation_loss_components(
        (torch.tensor(2.0), torch.tensor(0.75), torch.tensor(1.5), None)
    )

    assert total_loss.item() == 2.0
    assert pc_loss_sum.item() == 1.5


def test_visualization_history_matches_the_sampler_input_contract():
    point_clouds = torch.zeros(1, 3, 49, 1, 2048, 3)

    history = joint_simgen._visualization_history(point_clouds)

    assert history.shape == (3, 4, 2048, 3)


def test_4gpu_profile_has_four_processes_and_separate_output():
    accelerate_config = yaml.safe_load(
        Path("configs/accelerate/h200_4gpu.yaml").read_text()
    )
    training_config = yaml.safe_load(
        Path("configs/train/joint_simgen_480_4gpu.yaml").read_text()
    )

    assert accelerate_config["distributed_type"] == "MULTI_GPU"
    assert accelerate_config["num_processes"] == 4
    assert training_config["training"]["max_train_steps"] == 10_000
    assert training_config["logging"]["output_dir"] == "outputs/joint_simgen_4gpu"


def test_readme_documents_simgen_cache_preparation_and_training():
    readme = Path("README.md").read_text()

    assert "joint_simgen.py --prepare-utonia-cache" in readme
    assert "joint_simgen_480.yaml" in readme
