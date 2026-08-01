from pathlib import Path

import pytest
import torch

from train_joint_wan_physctrl import (
    combine_joint_losses,
    create_joint_optimizer,
    load_joint_checkpoint_with_fallback,
    prune_joint_checkpoints,
    should_log_denoised_latent_mse,
    should_save_joint_visualization,
    video_gradient_norm,
)


def test_joint_optimizer_uses_one_default_video_parameter_group():
    video = torch.nn.Linear(2, 2)
    pc = torch.nn.Linear(2, 2)
    bridge = torch.nn.Linear(2, 2)

    optimizer = create_joint_optimizer(
        list(video.parameters()) + list(pc.parameters()) + list(bridge.parameters())
    )

    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 1.0e-5
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1.0e-8
    assert group["weight_decay"] == 0.1


def test_joint_loss_adds_each_object_loss_without_averaging():
    video_loss = torch.tensor(2.0)
    object_losses = torch.tensor([[3.0, 5.0]])

    total = combine_joint_losses(video_loss, object_losses)

    assert total.item() == 10.0


def test_joint_visualization_runs_every_250_optimizer_steps():
    assert not should_save_joint_visualization(249, 250)
    assert should_save_joint_visualization(250, 250)


def test_joint_denoised_latent_mse_runs_every_50_optimizer_steps():
    assert not should_log_denoised_latent_mse(49, 50)
    assert should_log_denoised_latent_mse(50, 50)


def test_video_gradient_norm_uses_wan_dit_gradients_only():
    wan_parameter = torch.nn.Parameter(torch.zeros(2))
    pc_parameter = torch.nn.Parameter(torch.zeros(2))
    wan_parameter.grad = torch.tensor([3.0, 4.0])
    pc_parameter.grad = torch.tensor([100.0, 100.0])
    model = type(
        "Joint", (), {"wan_model": type("Wan", (), {"parameters": lambda self: [wan_parameter]})()}
    )()

    assert video_gradient_norm(model).item() == 5.0


class _FakeAccelerator:
    def __init__(self, failed_paths=()):
        self.failed_paths = {Path(path).name for path in failed_paths}
        self.attempts = []

    def load_state(self, path):
        self.attempts.append(Path(path).name)
        if Path(path).name in self.failed_paths:
            raise RuntimeError("incomplete checkpoint")


def test_joint_resume_latest_falls_back_from_incomplete_checkpoint(tmp_path):
    for step in (50, 100):
        (tmp_path / f"checkpoint-{step}").mkdir()
    accelerator = _FakeAccelerator({"checkpoint-100"})

    path = load_joint_checkpoint_with_fallback(accelerator, tmp_path, "latest")

    assert path.name == "checkpoint-50"
    assert accelerator.attempts == ["checkpoint-100", "checkpoint-50"]


def test_joint_checkpoint_pruning_keeps_the_newest_two(tmp_path):
    for step in (50, 100, 150):
        (tmp_path / f"checkpoint-{step}").mkdir()

    prune_joint_checkpoints(tmp_path, limit=2)

    assert {path.name for path in tmp_path.iterdir()} == {"checkpoint-100", "checkpoint-150"}
