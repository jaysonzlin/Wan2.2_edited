import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from train_pc import (
    build_pc_training_dataset,
    create_progress_bar,
    create_pc_noise_scheduler,
    first_unfinished_epoch,
    initialize_trackers,
    load_pc_checkpoint_with_fallback,
    prune_pc_checkpoints,
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


def test_ddpm_objective_creates_sample_prediction_scheduler():
    scheduler = create_pc_noise_scheduler(
        {"type": "ddpm", "num_train_timesteps": 1000, "beta_schedule": "linear"}
    )

    assert scheduler.config.prediction_type == "sample"
    assert scheduler.config.clip_sample is False
