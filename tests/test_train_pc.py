import subprocess
import sys
from pathlib import Path

from train_pc import (
    create_progress_bar,
    create_pc_noise_scheduler,
    initialize_trackers,
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


def test_visualization_cadence_uses_completed_epochs():
    assert not should_save_visualization(epoch=1, every_epochs=2)
    assert should_save_visualization(epoch=2, every_epochs=2)
    assert should_save_visualization(epoch=3, every_epochs=3)


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


def test_ddpm_objective_creates_sample_prediction_scheduler():
    scheduler = create_pc_noise_scheduler(
        {"type": "ddpm", "num_train_timesteps": 1000, "beta_schedule": "linear"}
    )

    assert scheduler.config.prediction_type == "sample"
    assert scheduler.config.clip_sample is False
