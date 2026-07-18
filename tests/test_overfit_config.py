import tempfile
import unittest
from pathlib import Path

import yaml

from training.overfit_config import load_config


def write_yaml(directory: Path, data: dict) -> Path:
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class OverfitConfigTests(unittest.TestCase):
    def test_dotlist_overrides_win_over_yaml(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_yaml(
                Path(temporary_directory), {"training": {"max_train_steps": 5000}}
            )
            config = load_config(path, ["training.max_train_steps=3"])
            self.assertEqual(config["training"]["max_train_steps"], 3)

    def test_invalid_temporal_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_yaml(
                Path(temporary_directory), {"data": {"num_frames": 48}}
            )
            with self.assertRaisesRegex(ValueError, "4n \\+ 1"):
                load_config(path, [])
