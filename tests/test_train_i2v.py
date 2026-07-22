import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from train_i2v import (
    create_progress_bar,
    load_checkpoint_with_fallback,
    prune_checkpoints,
    visualization_path,
)


class TrainI2VHelperTests(unittest.TestCase):
    def test_latest_checkpoint_falls_back_after_a_failed_load(self):
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

            resumed_path = load_checkpoint_with_fallback(accelerator, root, "latest")

        self.assertEqual(resumed_path.name, "checkpoint-500")
        self.assertEqual(accelerator.attempts, ["checkpoint-750", "checkpoint-500"])

    def test_latest_checkpoint_reports_error_after_all_loads_fail(self):
        class FakeAccelerator:
            def load_state(self, path):
                raise RuntimeError(f"incomplete {Path(path).name}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in (250, 500):
                (root / f"checkpoint-{step}").mkdir()

            with self.assertRaisesRegex(RuntimeError, "checkpoint-500, checkpoint-250"):
                load_checkpoint_with_fallback(FakeAccelerator(), root, "latest")

    def test_progress_bar_tracks_optimizer_steps(self):
        progress_bar = create_progress_bar(total=5000, initial=12, enabled=True)
        try:
            self.assertEqual(progress_bar.total, 5000)
            self.assertEqual(progress_bar.n, 12)
        finally:
            progress_bar.close()

    def test_help_does_not_import_remote_only_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "train_i2v.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)

    def test_visualization_name_uses_epoch(self):
        self.assertEqual(
            visualization_path(Path("outputs"), 12),
            Path("outputs") / "vis" / "epoch_0012.mp4",
        )

    def test_visualization_metrics_are_logged_only_with_the_visualization(self):
        source = Path("train_i2v.py").read_text()
        compact_source = "".join(source.split())

        self.assertIn("return latent", source)
        self.assertIn(
            "denoised_latent_mse(visualization_latent,clean_latents[0])",
            compact_source,
        )
        self.assertIn('"train/visualization_denoised_latent_mse"', source)

    def test_checkpoint_pruning_keeps_newest_three(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in (250, 500, 750, 1000):
                (root / f"checkpoint-{step}").mkdir()

            prune_checkpoints(root, 3)

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"checkpoint-500", "checkpoint-750", "checkpoint-1000"},
            )
