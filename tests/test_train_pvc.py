import torch

from train_pvc import build_pvc_training_dataset, compute_pvc_training_prediction


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
