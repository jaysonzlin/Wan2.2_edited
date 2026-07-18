import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from train_i2v import prune_checkpoints, visualization_path


class TrainI2VHelperTests(unittest.TestCase):
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
