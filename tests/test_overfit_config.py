import tempfile
import unittest
from pathlib import Path

import yaml
import torch

from training.overfit_config import load_config
from training.schedules import create_lr_scheduler


def write_yaml(directory: Path, data: dict) -> Path:
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class OverfitConfigTests(unittest.TestCase):
    def test_constant_scheduler_uses_the_constant_warmup_factory(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))])
        calls = []

        def cosine_factory(*args):
            calls.append(("cosine", args))
            return "cosine"

        def constant_factory(*args):
            calls.append(("constant", args))
            return "constant"

        scheduler = create_lr_scheduler(
            "constant", optimizer, warmup_steps=200, max_train_steps=5000,
            cosine_factory=cosine_factory, constant_factory=constant_factory,
        )

        self.assertEqual(scheduler, "constant")
        self.assertEqual(calls, [("constant", (optimizer, 200))])

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

    def test_invalid_denoised_latent_mse_cadence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_yaml(
                Path(temporary_directory),
                {"training": {"denoised_latent_mse_every_steps": 0}},
            )

            with self.assertRaisesRegex(
                ValueError, "denoised_latent_mse_every_steps must be a positive integer"
            ):
                load_config(path, [])
