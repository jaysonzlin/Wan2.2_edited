import pytest
import torch

from train_pvc import (
    build_pvc_lr_scheduler,
    build_pvc_training_model,
    build_pvc_training_dataset,
    compute_pvc_training_prediction,
    mean_position_error,
    pvc_loss,
    sample_pvc_visualization,
)


def test_pvc_training_model_forwards_the_selected_view_gate_mode():
    seen = {}

    model = build_pvc_training_model(
        {"model": {"point_view_gate_mode": "separate"}},
        5,
        model_factory=lambda **kwargs: seen.update(kwargs) or "model",
    )

    assert model == "model"
    assert seen["point_view_gate_mode"] == "separate"


def test_pvc_training_model_defaults_to_shared_view_gate_mode():
    seen = {}

    build_pvc_training_model(
        {"model": {}}, 5, model_factory=lambda **kwargs: seen.update(kwargs)
    )

    assert seen["point_view_gate_mode"] == "shared"


def test_pvc_loss_adds_weighted_object_centroid_position_error():
    prediction = torch.zeros(1, 2, 1, 2, 3)
    target = torch.tensor([3.0, 4.0, 0.0]).reshape(1, 1, 1, 1, 3).expand(
        1, 2, 1, 2, 3
    )

    assert mean_position_error(prediction, target).item() == 5.0
    assert pvc_loss(prediction, target, position_loss_weight=0.5).item() == pytest.approx(
        65.0 / 6.0
    )


def test_pvc_lr_scheduler_provides_required_constant_factory():
    optimizer = object()
    calls = []

    result = build_pvc_lr_scheduler(
        "constant", optimizer, 3, 9,
        cosine_factory=lambda *_args: None,
        constant_factory=lambda passed_optimizer, warmup: calls.append((passed_optimizer, warmup)) or "scheduler",
    )

    assert result == "scheduler"
    assert calls == [(optimizer, 3)]


def test_pvc_dataset_builder_prepares_both_caches_in_feature_width_order():
    calls = []

    class Dataset:
        source_paths = {"sample": "pc"}
        point_view_source_paths = {"sample": ("view",) * 49}

        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs

    class Extractor:
        pass

    dataset, width = build_pvc_training_dataset(
        {"data": {"dataset_root": "root", "object_id": "000", "utonia_cache_root": "pc-cache", "point_view_utonia_cache_root": "view-cache"}, "seed": 7},
        dataset_factory=Dataset, extractor_factory=lambda *_args: Extractor(),
        trajectory_cache_preparer=lambda sources, root, extractor: calls.append(("pc", sources, root)) or 5,
        point_view_cache_preparer=lambda sources, root, extractor, *, feature_dim: calls.append(("views", sources, root, feature_dim)),
    )

    assert width == 5
    assert calls == [("pc", {"sample": "pc"}, "pc-cache"), ("views", {"sample": ("view",) * 49}, "view-cache", 5)]
    assert dataset.kwargs["point_view_feature_dim"] == 5


def test_pvc_training_prediction_forwards_every_condition():
    future = torch.zeros(1, 45, 1, 2, 3)
    history = torch.zeros(1, 4, 1, 2, 3)
    views = torch.zeros(1, 49, 2, 3)
    mask = torch.ones(1, 49, 2, dtype=torch.bool)
    features = torch.zeros(1, 2, 5)
    view_features = torch.zeros(1, 49, 2, 5)
    seen = []
    batch = type("Batch", (), {"model_input": torch.ones_like(future), "frame_times": torch.zeros(1, 49), "target": future})()

    prediction, target = compute_pvc_training_prediction(
        {"points_tgt": future, "points_history": history, "point_views": views, "point_view_mask": mask, "utonia_features": features, "point_view_utonia_features": view_features},
        lambda *args: seen.append(args) or torch.full_like(future, 2), "scheduler", torch.Generator(), "cpu",
        ddpm_batch_factory=lambda *_args, **_kwargs: batch,
    )

    assert torch.equal(prediction, torch.full_like(future, 2))
    assert target is future
    assert seen[0][2:] == (history, views, mask, features, view_features)


def test_pvc_visualization_prepends_history_and_forwards_view_conditions():
    history = torch.randn(1, 4, 1, 2, 3)
    future = torch.randn(1, 45, 1, 2, 3)
    views = torch.randn(1, 49, 2, 3)
    mask = torch.ones(1, 49, 2, dtype=torch.bool)
    features = torch.randn(1, 2, 5)
    view_features = torch.randn(1, 49, 2, 5)
    calls = []

    def pipeline(*args):
        calls.append(args)
        return torch.zeros_like(future)

    predicted, ground_truth = sample_pvc_visualization(
        pipeline,
        {"points_history": history, "points_tgt": future, "point_views": views,
         "point_view_mask": mask, "utonia_features": features,
         "point_view_utonia_features": view_features},
        "cpu", 5, torch.Generator(),
    )

    assert predicted.shape == ground_truth.shape == (49, 1, 2, 3)
    assert torch.equal(predicted[:4], history.squeeze(0))
    assert calls[0][1:3] == (views, mask)
