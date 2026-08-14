import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from train_pc import (
    build_pc_training_model,
    build_pc_training_dataset,
    build_pc_sampling_pipeline,
    compute_pc_training_prediction,
    create_progress_bar,
    create_pc_noise_scheduler,
    first_unfinished_epoch,
    initialize_trackers,
    load_pc_checkpoint_with_fallback,
    prune_pc_checkpoints,
    sample_pc_visualization,
    should_save_visualization,
    visualization_path,
)


def test_train_pc_help_is_local_only():
    result = subprocess.run(
        [sys.executable, "train_pc.py", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_visualization_path_is_inside_configured_vis_directory():
    assert visualization_path(Path("outputs/run"), "vis", 12) == Path("outputs/run/vis/epoch_0012.mp4")


def test_progress_bar_tracks_optimizer_steps():
    progress_bar = create_progress_bar(total=10, initial=2, enabled=True)
    try:
        assert progress_bar.total == 10
        assert progress_bar.n == 2
    finally:
        progress_bar.close()


def test_train_pc_restores_step_after_accelerator_prepare():
    source = Path("train_pc.py").read_text()

    prepared = source.index("accelerator.prepare")
    restored = source.index("load_pc_checkpoint_with_fallback", prepared)
    assert restored > prepared
    assert 'step = int(resume_path.name.removeprefix("checkpoint-"))' in source
    assert "initial=step" in source


def test_pc_latest_checkpoint_falls_back_after_a_failed_load():
    class FakeAccelerator:
        def __init__(self):
            self.attempts = []

        def load_state(self, path):
            self.attempts.append(Path(path).name)
            if Path(path).name == "checkpoint-750":
                raise RuntimeError("incomplete checkpoint")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for step in (250, 500, 750):
            (root / f"checkpoint-{step}").mkdir()
        accelerator = FakeAccelerator()

        resumed = load_pc_checkpoint_with_fallback(accelerator, root, "latest")

    assert resumed.name == "checkpoint-500"
    assert accelerator.attempts == ["checkpoint-750", "checkpoint-500"]


def test_pc_explicit_checkpoint_propagates_load_failure(tmp_path):
    checkpoint = tmp_path / "checkpoint-250"
    checkpoint.mkdir()

    class FakeAccelerator:
        def load_state(self, _path):
            raise RuntimeError("incomplete checkpoint")

    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        load_pc_checkpoint_with_fallback(FakeAccelerator(), tmp_path, str(checkpoint))


def test_pc_latest_checkpoint_reports_all_failed_candidates(tmp_path):
    for step in (250, 500):
        (tmp_path / f"checkpoint-{step}").mkdir()

    class FakeAccelerator:
        def load_state(self, path):
            raise RuntimeError(f"incomplete {Path(path).name}")

    with pytest.raises(RuntimeError, match="checkpoint-500, checkpoint-250"):
        load_pc_checkpoint_with_fallback(FakeAccelerator(), tmp_path, "latest")


def _write_history_model_checkpoint(checkpoint):
    from wan.modules.pc_trajectory import PCTrajectoryModel

    model = build_pc_training_model(
        _tiny_model_config("history"), None, model_factory=PCTrajectoryModel
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 100)
    checkpoint.mkdir()
    torch.save(model.state_dict(), checkpoint / "model.pt")
    return model


class _ModelStateAccelerator:
    def __init__(self, model):
        self.model = model

    def load_state(self, checkpoint):
        state = torch.load(Path(checkpoint) / "model.pt", weights_only=True)
        self.model.load_state_dict(state, strict=True)


def test_history_model_resumes_from_latest_same_mode_checkpoint(tmp_path):
    from wan.modules.pc_trajectory import PCTrajectoryModel

    source = _write_history_model_checkpoint(tmp_path / "checkpoint-7")
    resumed_model = build_pc_training_model(
        _tiny_model_config("history"), None, model_factory=PCTrajectoryModel
    )

    resumed = load_pc_checkpoint_with_fallback(
        _ModelStateAccelerator(resumed_model), tmp_path, "latest"
    )

    assert resumed == tmp_path / "checkpoint-7"
    assert all(
        torch.equal(resumed_model.state_dict()[key], value)
        for key, value in source.state_dict().items()
    )


def test_history_checkpoint_rejects_velocity_model_layout(tmp_path):
    from wan.modules.pc_trajectory import PCTrajectoryModel

    checkpoint = tmp_path / "checkpoint-7"
    _write_history_model_checkpoint(checkpoint)
    velocity_model = build_pc_training_model(
        _tiny_model_config(), None, model_factory=PCTrajectoryModel
    )

    with pytest.raises(RuntimeError, match="state_dict"):
        load_pc_checkpoint_with_fallback(
            _ModelStateAccelerator(velocity_model), tmp_path, str(checkpoint)
        )


def test_prune_pc_checkpoints_keeps_only_latest_numeric_states():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for name in (
            "checkpoint-250",
            "checkpoint-500",
            "checkpoint-750",
            "checkpoint-draft",
        ):
            (root / name).mkdir()

        prune_pc_checkpoints(root, 2)

        assert {path.name for path in root.iterdir()} == {
            "checkpoint-500",
            "checkpoint-750",
            "checkpoint-draft",
        }


def test_visualization_cadence_uses_completed_epochs():
    assert not should_save_visualization(epoch=1, every_epochs=2)
    assert should_save_visualization(epoch=2, every_epochs=2)
    assert should_save_visualization(epoch=3, every_epochs=3)


def test_resume_starts_visualization_epochs_after_the_saved_single_sample_step():
    assert first_unfinished_epoch(completed_steps=250) == 251


def test_initialize_trackers_uses_the_configured_wandb_project():
    class FakeAccelerator:
        def __init__(self):
            self.calls = []

        def init_trackers(self, project_name, config):
            self.calls.append((project_name, config))

    config = {"tracker_project_name": "pc_flow", "report_to": "wandb"}
    accelerator = FakeAccelerator()

    initialize_trackers(accelerator, config)

    assert accelerator.calls == [("pc_flow", config)]


def test_readme_documents_pc_flow_entrypoint():
    readme = Path("README.md").read_text()

    assert "train_pc.py --config configs/train/config_pc.yaml" in readme
    assert "pc.hdf5" in readme


def test_readme_documents_utonia_pc_overfit_entrypoint():
    readme = Path("README.md").read_text()

    assert "train_pc.py --config configs/train/config_pc_utonia_overfit.yaml" in readme
    assert "vendored Utonia inference runtime" in readme
    assert "install the sibling Utonia package" not in readme


def test_utonia_config_is_loadable():
    from training.pc_config import load_pc_config

    config = load_pc_config("configs/train/config_pc_utonia_overfit.yaml", [])

    assert config["model"]["utonia_enabled"] is True
    assert config["data"]["object_id"] == "000"


def test_utonia_history_config_loads_a_distinct_four_frame_experiment():
    from training.pc_config import load_pc_config

    config = load_pc_config("configs/train/config_pc_utonia_history_overfit.yaml", [])

    assert config["output_dir"] == "./outputs/pc_trajectory_utonia_history_overfit"
    assert config["tracker_project_name"] == "pc_trajectory_utonia_history_overfit"
    assert config["resume_from_checkpoint"] is None
    assert config["model"]["conditioning"] == "history"
    assert config["model"]["history_frames"] == 4
    assert config["model"]["utonia_enabled"] is True


def test_build_pc_training_dataset_prepares_cache_before_cache_backed_dataset():
    calls = []

    class Dataset:
        source_paths = {"sample_0/objects/000": Path("input/pc.hdf5")}

    def dataset_factory(*args, **kwargs):
        calls.append(("dataset", args, kwargs))
        return Dataset()

    def extractor_factory(cache_root):
        calls.append(("extractor", cache_root))
        return "extractor"

    def cache_preparer(sources, cache_root, extractor):
        calls.append(("prepare", sources, cache_root, extractor))
        return 17

    config = {
        "data": {
            "dataset_root": "input",
            "object_id": "000",
            "utonia_cache_root": "utonia-cache",
        },
        "model": {"utonia_enabled": True},
    }

    dataset, feature_dim = build_pc_training_dataset(
        config,
        dataset_factory=dataset_factory,
        extractor_factory=extractor_factory,
        cache_preparer=cache_preparer,
    )

    assert isinstance(dataset, Dataset)
    assert feature_dim == 17
    assert calls == [
        ("dataset", ("input",), {"object_id": "000"}),
        ("extractor", "utonia-cache"),
        (
            "prepare",
            {"sample_0/objects/000": Path("input/pc.hdf5")},
            "utonia-cache",
            "extractor",
        ),
        (
            "dataset",
            ("input",),
            {"object_id": "000", "utonia_cache_root": "utonia-cache"},
        ),
    ]


def test_build_pc_training_dataset_keeps_baseline_path_cache_free():
    calls = []

    def dataset_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "baseline"

    dataset, feature_dim = build_pc_training_dataset(
        {"data": {"dataset_root": "input"}, "model": {}},
        dataset_factory=dataset_factory,
        extractor_factory=lambda _root: pytest.fail("extractor should not be built"),
        cache_preparer=lambda *_args: pytest.fail("cache should not be prepared"),
    )

    assert (dataset, feature_dim) == ("baseline", None)
    assert calls == [(("input",), {})]


def test_build_pc_training_dataset_uses_object_dataset_without_utonia():
    calls = []

    def dataset_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "object-dataset"

    dataset, feature_dim = build_pc_training_dataset(
        {
            "data": {"dataset_root": "input", "object_id": "000"},
            "model": {"utonia_enabled": False},
        },
        dataset_factory=dataset_factory,
        extractor_factory=lambda _root: pytest.fail("extractor should not be built"),
        cache_preparer=lambda *_args: pytest.fail("cache should not be prepared"),
    )

    assert (dataset, feature_dim) == ("object-dataset", None)
    assert calls == [(("input",), {"object_id": "000"})]


def test_build_pc_training_dataset_forwards_history_only_in_history_mode():
    calls = []

    class Dataset:
        source_paths = {"sample_0/objects/000": Path("input/pc.hdf5")}

    def dataset_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset()

    config = {
        "data": {
            "dataset_root": "input",
            "object_id": "000",
            "utonia_cache_root": "utonia-cache",
        },
        "model": {
            "conditioning": "history",
            "history_frames": 4,
            "utonia_enabled": True,
        },
    }

    build_pc_training_dataset(
        config,
        dataset_factory=dataset_factory,
        extractor_factory=lambda _root: "extractor",
        cache_preparer=lambda *_args: 17,
    )

    assert calls == [
        (("input",), {"object_id": "000", "history_frames": 4}),
        (
            ("input",),
            {
                "object_id": "000",
                "utonia_cache_root": "utonia-cache",
                "history_frames": 4,
            },
        ),
    ]


def _tiny_model_config(conditioning=None):
    model = {
        "latent_dim": 64,
        "n_layers": 1,
        "num_heads": 1,
        "point_embed": True,
    }
    if conditioning is not None:
        model.update(conditioning=conditioning, history_frames=4)
    return {
        "data": {"num_points": 8},
        "model": model,
        "objective": {"type": "ddpm"},
    }


def test_build_pc_training_model_configures_four_frame_history():
    calls = []

    def model_factory(**kwargs):
        calls.append(kwargs)
        return "model"

    model = build_pc_training_model(
        _tiny_model_config("history"), 17, model_factory=model_factory
    )

    assert model == "model"
    assert calls == [
        {
            "n_points": 8,
            "n_future_frames": 48,
            "latent_dim": 64,
            "n_layers": 1,
            "num_heads": 1,
            "point_embed": True,
            "objective_type": "ddpm",
            "utonia_feature_dim": 17,
            "conditioning": "history",
            "history_frames": 4,
        }
    ]


def test_build_pc_training_model_preserves_velocity_factory_call():
    calls = []

    def legacy_model_factory(
        *,
        n_points,
        n_future_frames,
        latent_dim,
        n_layers,
        num_heads,
        point_embed,
        objective_type,
        utonia_feature_dim,
    ):
        calls.append(
            (
                n_points,
                n_future_frames,
                latent_dim,
                n_layers,
                num_heads,
                point_embed,
                objective_type,
                utonia_feature_dim,
            )
        )
        return "legacy-model"

    model = build_pc_training_model(
        _tiny_model_config(), None, model_factory=legacy_model_factory
    )

    assert model == "legacy-model"
    assert calls == [(8, 48, 64, 1, 1, True, "ddpm", None)]


def test_history_training_prediction_uses_history_without_velocity_tensors():
    history = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 1, 2, 3)
    future = torch.zeros(1, 45, 1, 2, 3)
    features = torch.randn(1, 2, 5)
    model_input = torch.ones_like(future)
    frame_times = torch.cat((torch.zeros(1, 4), torch.full((1, 45), 12.0)), dim=1)
    target = torch.full_like(future, 3.0)
    factory_calls = []
    model_calls = []

    def ddpm_batch_factory(points, scheduler, generator, known_frames):
        factory_calls.append((points, scheduler, generator, known_frames))
        return type(
            "Batch",
            (),
            {"model_input": model_input, "frame_times": frame_times, "target": target},
        )()

    def history_model(noisy, times, known, *, utonia_features=None):
        model_calls.append((noisy, times, known, utonia_features))
        return torch.full_like(noisy, 6.0)

    generator = torch.Generator().manual_seed(0)
    prediction, returned_target = compute_pc_training_prediction(
        {
            "points_history": history,
            "points_tgt": future,
            "initial_linear_velocity": torch.full((1, 1, 3), 91.0),
            "initial_angular_velocity": torch.full((1, 1, 3), 92.0),
            "utonia_features": features,
        },
        history_model,
        {"type": "ddpm"},
        "scheduler",
        generator,
        "cpu",
        "history",
        flow_batch_factory=lambda *_args: pytest.fail("flow batch should not be used"),
        ddpm_batch_factory=ddpm_batch_factory,
    )

    assert torch.equal(prediction, torch.full_like(future, 6.0))
    assert returned_target is target
    assert factory_calls == [(future, "scheduler", generator, 4)]
    assert len(model_calls) == 1
    assert model_calls[0][0] is model_input
    assert model_calls[0][1] is frame_times
    assert torch.equal(model_calls[0][2], history)
    assert model_calls[0][3] is features


def test_velocity_training_prediction_preserves_positional_flow_calls():
    source = torch.zeros(1, 1, 2, 3)
    future = torch.zeros(1, 48, 1, 2, 3)
    linear = torch.full((1, 1, 3), 1.0)
    angular = torch.full((1, 1, 3), 2.0)
    model_input = torch.ones_like(future)
    frame_times = torch.ones(1, 49)
    target = torch.full_like(future, 3.0)
    factory_calls = []
    model_calls = []

    def legacy_flow_batch_factory(
        points, initial, generator, time_shift, num_train_timesteps
    ):
        factory_calls.append(
            (points, initial, generator, time_shift, num_train_timesteps)
        )
        return type(
            "Batch",
            (),
            {
                "model_input": model_input,
                "frame_times": frame_times,
                "velocity_target": target,
            },
        )()

    def legacy_model(
        noisy,
        times,
        initial,
        initial_linear,
        initial_angular,
        *,
        utonia_features=None,
    ):
        model_calls.append(
            (noisy, times, initial, initial_linear, initial_angular, utonia_features)
        )
        return torch.full_like(noisy, 6.0)

    generator = torch.Generator().manual_seed(0)
    prediction, returned_target = compute_pc_training_prediction(
        {
            "points_src": source,
            "points_tgt": future,
            "initial_linear_velocity": linear,
            "initial_angular_velocity": angular,
        },
        legacy_model,
        {"type": "flow", "time_shift": 5.0, "num_train_timesteps": 1000},
        None,
        generator,
        "cpu",
        "velocity",
        flow_batch_factory=legacy_flow_batch_factory,
        ddpm_batch_factory=lambda *_args: pytest.fail("DDPM batch should not be used"),
    )

    assert torch.equal(prediction, torch.full_like(future, 6.0))
    assert returned_target is target
    assert factory_calls == [(future, source, generator, 5.0, 1000)]
    assert len(model_calls) == 1
    assert model_calls[0][:5] == (model_input, frame_times, source, linear, angular)
    assert model_calls[0][5] is None


def test_history_visualization_prepends_all_four_known_frames_and_forwards_utonia():
    history = torch.stack(
        [torch.full((1, 1, 2, 3), value) for value in (1.0, 2.0, 3.0, 4.0)],
        dim=1,
    )
    future = torch.full((1, 45, 1, 2, 3), 8.0)
    predicted_future = torch.full_like(future, 9.0)
    features = torch.randn(1, 2, 5)
    calls = []

    def history_pipeline(known, device, steps, generator, *, utonia_features=None):
        calls.append((known, device, steps, generator, utonia_features))
        return predicted_future

    generator = torch.Generator().manual_seed(0)
    predicted, ground_truth = sample_pc_visualization(
        history_pipeline,
        {"points_history": history, "points_tgt": future, "utonia_features": features},
        "history",
        "cpu",
        7,
        generator,
    )

    assert calls == [(history, "cpu", 7, generator, features)]
    assert predicted.shape == ground_truth.shape == (49, 1, 2, 3)
    assert torch.equal(predicted[:4], history.squeeze(0))
    assert torch.equal(predicted[4:], predicted_future.squeeze(0))
    assert torch.equal(ground_truth[:4], history.squeeze(0))
    assert torch.equal(ground_truth[4:], future.squeeze(0))


def test_velocity_visualization_preserves_legacy_pipeline_call():
    source = torch.full((1, 1, 2, 3), 1.0)
    future = torch.full((1, 48, 1, 2, 3), 8.0)
    predicted_future = torch.full_like(future, 9.0)
    linear = torch.full((1, 1, 3), 2.0)
    angular = torch.full((1, 1, 3), 3.0)
    calls = []

    def legacy_pipeline(
        initial,
        initial_linear,
        initial_angular,
        device,
        steps,
        generator,
        *,
        utonia_features=None,
    ):
        calls.append(
            (
                initial,
                initial_linear,
                initial_angular,
                device,
                steps,
                generator,
                utonia_features,
            )
        )
        return predicted_future

    generator = torch.Generator().manual_seed(0)
    predicted, ground_truth = sample_pc_visualization(
        legacy_pipeline,
        {
            "points_src": source,
            "points_tgt": future,
            "initial_linear_velocity": linear,
            "initial_angular_velocity": angular,
        },
        "velocity",
        "cpu",
        7,
        generator,
    )

    assert calls == [(source, linear, angular, "cpu", 7, generator, None)]
    assert predicted.shape == ground_truth.shape == (49, 1, 2, 3)
    assert torch.equal(predicted[:1], source)
    assert torch.equal(predicted[1:], predicted_future.squeeze(0))


@pytest.mark.parametrize(
    ("objective_type", "conditioning", "expected_factory", "expected_kwargs"),
    [
        ("flow", "velocity", "flow", {"time_shift": 5.0}),
        ("flow", "history", "history-flow", {"time_shift": 5.0}),
        ("ddpm", "velocity", "ddim", {}),
        ("ddpm", "history", "history-ddim", {}),
    ],
)
def test_sampling_pipeline_selection_matches_objective_and_conditioning(
    objective_type, conditioning, expected_factory, expected_kwargs
):
    calls = []

    def factory(name):
        def build(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return build

    pipeline = build_pc_sampling_pipeline(
        "model",
        "scheduler",
        objective_type,
        conditioning,
        time_shift=5.0,
        flow_pipeline_factory=factory("flow"),
        history_flow_pipeline_factory=factory("history-flow"),
        ddim_pipeline_factory=factory("ddim"),
        history_ddim_pipeline_factory=factory("history-ddim"),
    )

    assert pipeline == expected_factory
    assert calls == [
        (expected_factory, ("model", "scheduler"), expected_kwargs)
    ]


def test_ddpm_objective_creates_sample_prediction_scheduler():
    scheduler = create_pc_noise_scheduler(
        {"type": "ddpm", "num_train_timesteps": 1000, "beta_schedule": "linear"}
    )

    assert scheduler.config.prediction_type == "sample"
    assert scheduler.config.clip_sample is False
