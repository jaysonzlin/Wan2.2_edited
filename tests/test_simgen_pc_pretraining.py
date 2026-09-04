import json
from pathlib import Path

import pretrain_simgen_pc
import torch
from diffusers import DDPMScheduler

from training.simgen_pc_pretraining import (
    flatten_simgen_pc_batch,
    load_best_export_metadata,
    object_mse_totals,
    pc_pretraining_prediction,
    reduce_object_mean,
    save_best_pc_export,
    save_sample_zero_visualizations,
)


class RecordingModel:
    def __call__(self, noisy_future, frame_times, history, *_unused, utonia_features):
        self.noisy_future = noisy_future
        self.frame_times = frame_times
        self.history = history
        self.utonia_features = utonia_features
        return torch.zeros_like(noisy_future)


class SumReducer:
    def __init__(self, values):
        self.values = iter(values)

    def reduce(self, _value, reduction):
        assert reduction == "sum"
        return next(self.values)


def _two_object_batch():
    point_clouds = torch.arange(2 * 49 * 1 * 2 * 3, dtype=torch.float32).reshape(
        1, 2, 49, 1, 2, 3
    )
    utonia_features = torch.arange(2 * 2 * 5, dtype=torch.float32).reshape(1, 2, 2, 5)
    return {"point_clouds": point_clouds, "utonia_features": utonia_features}


def test_flatten_simgen_objects_preserves_each_history_future_and_feature():
    batch = _two_object_batch()

    flattened = flatten_simgen_pc_batch(batch, device="cpu")

    assert flattened.history.shape == (2, 4, 1, 2, 3)
    assert flattened.future.shape == (2, 45, 1, 2, 3)
    assert torch.equal(flattened.history[1], batch["point_clouds"][0, 1, :4])
    assert torch.equal(flattened.future[0], batch["point_clouds"][0, 0, 4:])
    assert torch.equal(flattened.utonia_features[0], batch["utonia_features"][0, 0])


def test_pc_pretraining_prediction_uses_history_and_clean_future_target():
    flattened = flatten_simgen_pc_batch(_two_object_batch(), device="cpu")
    model = RecordingModel()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    prediction, target = pc_pretraining_prediction(
        flattened, model, scheduler, torch.Generator().manual_seed(0), "cpu"
    )

    assert prediction.shape == target.shape == (2, 45, 1, 2, 3)
    assert torch.equal(target, flattened.future)
    assert torch.equal(model.history, flattened.history)
    assert torch.equal(model.utonia_features, flattened.utonia_features)
    assert model.frame_times.shape == (2, 49)
    assert torch.equal(model.frame_times[:, :4], torch.zeros(2, 4))


def test_object_mse_totals_and_reduction_weight_each_object_equally():
    prediction = torch.tensor([[[[1.0]]], [[[3.0]]]])
    target = torch.tensor([[[[0.0]]], [[[1.0]]]])

    local_sum, local_count = object_mse_totals(prediction, target)
    reduced = reduce_object_mean(
        SumReducer([torch.tensor(14.0), torch.tensor(4.0)]), local_sum, local_count
    )

    assert local_sum.item() == 5.0
    assert local_count == 2
    assert reduced.item() == 3.5


def test_best_export_replaces_only_a_lower_validation_loss(tmp_path):
    model = torch.nn.Linear(2, 1)

    assert save_best_pc_export(model, tmp_path, validation_loss=0.4, step=100, best_loss=float("inf"))
    assert not save_best_pc_export(model, tmp_path, validation_loss=0.5, step=200, best_loss=0.4)

    assert load_best_export_metadata(tmp_path) == (0.4, 100)
    assert json.loads((tmp_path / "best_pc_model.json").read_text()) == {
        "validation_loss": 0.4,
        "step": 100,
    }
    assert (tmp_path / "best_pc_model.pt").is_file()


def test_sample_zero_visualization_writes_one_comparison_per_object(monkeypatch, tmp_path):
    class FakeHistoryPipeline:
        def __call__(self, history, device, num_inference_steps, generator, *, utonia_features):
            assert device == "cpu"
            assert num_inference_steps == 7
            assert isinstance(generator, torch.Generator)
            assert torch.equal(utonia_features, flattened.utonia_features)
            return torch.zeros_like(flattened.future)

    flattened = flatten_simgen_pc_batch(_two_object_batch(), device="cpu")
    saved = []
    monkeypatch.setattr(
        "training.simgen_pc_pretraining.save_pointcloud_comparison_mp4",
        lambda prediction, ground_truth, path, fps: saved.append(
            (prediction.shape, ground_truth.shape, path, fps)
        ),
    )

    save_sample_zero_visualizations(
        FakeHistoryPipeline(),
        flattened,
        tmp_path,
        step=1000,
        fps=12,
        device="cpu",
        num_inference_steps=7,
        generator=torch.Generator().manual_seed(0),
    )

    assert [path.name for _, _, path, _ in saved] == [
        "object_000_trajectory_comparison.mp4",
        "object_001_trajectory_comparison.mp4",
    ]
    assert all(shape == (49, 1, 2, 3) for shape, _, _, _ in saved)
    assert all(fps == 12 for _, _, _, fps in saved)


def test_pretraining_builds_the_configured_simgen_train_and_validation_splits(monkeypatch):
    calls = []

    class Dataset:
        def __init__(self, root, sample_ids, **kwargs):
            calls.append((root, sample_ids, kwargs))

    monkeypatch.setattr(pretrain_simgen_pc, "SimGenJointDataset", Dataset)
    config = {
        "data": {
            "dataset_root": "simgen",
            "train_start": 0,
            "train_end": 127,
            "validation_start": 490,
            "validation_end": 499,
            "num_points": 2048,
            "utonia_cache_root": "cache",
        }
    }

    pretrain_simgen_pc.build_datasets(config)

    assert calls[0][1] == list(range(128))
    assert calls[1][1] == list(range(490, 500))
    assert calls[0][2]["utonia_cache_root"] == "cache"


def test_pretraining_launcher_uses_four_gpus_and_latest_resume():
    source = Path("submit_pretrain_simgen_pc_4gpu.sh").read_text()

    assert "#SBATCH --nodes=1" in source
    assert "configs/accelerate/h200_4gpu.yaml" in source
    assert "pretrain_simgen_pc.py" in source
    assert "training.resume_from_checkpoint=latest" in source
