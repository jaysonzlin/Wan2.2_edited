from pathlib import Path

import pytest
import torch

from train_joint_wan_physctrl import (
    add_rigid_metrics,
    combine_joint_losses,
    create_joint_optimizer,
    load_joint_checkpoint_with_fallback,
    pc_gradient_norm,
    per_object_metric_values,
    prune_joint_checkpoints,
    rigid_loss_terms,
    should_log_denoised_latent_mse,
    should_save_joint_visualization,
    video_gradient_norm,
)


def _optimizer_config():
    return {
        "video": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "bca": {"lr": 1.0e-5, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.1},
        "pc": {"lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 1.0e-2},
    }


def test_joint_optimizer_uses_separate_configured_parameter_groups():
    model = type("Joint", (), {
        "wan_model": torch.nn.Linear(2, 2),
        "bridges": torch.nn.ModuleList([torch.nn.Linear(2, 2)]),
        "pc_model": torch.nn.Linear(2, 2),
    })()

    optimizer = create_joint_optimizer(model, _optimizer_config())

    assert [group["name"] for group in optimizer.param_groups] == ["video", "bca", "pc"]
    assert optimizer.param_groups[0]["lr"] == 1.0e-5
    assert optimizer.param_groups[1]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[2]["lr"] == 1.0e-4
    assert optimizer.param_groups[2]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[2]["eps"] == 1.0e-8
    assert optimizer.param_groups[2]["weight_decay"] == 1.0e-2


def test_joint_loss_adds_each_object_loss_without_averaging():
    video_loss = torch.tensor(2.0)
    object_losses = torch.tensor([[3.0, 5.0]])

    total = combine_joint_losses(
        video_loss,
        object_losses,
        rigid_loss_sum=torch.tensor(0.0),
        rigid_loss_weight=0.0,
    )

    assert total.item() == 10.0


def test_joint_loss_adds_weighted_rigid_sum_without_object_averaging():
    total = combine_joint_losses(
        video_loss=torch.tensor(2.0),
        object_losses=torch.tensor([[3.0, 5.0]]),
        rigid_loss_sum=torch.tensor(7.0),
        rigid_loss_weight=0.25,
    )

    assert total.item() == 11.75


def test_per_object_metric_values_uses_zero_padded_slots():
    metrics = per_object_metric_values(
        "train/rigid_loss_object", torch.tensor([1.5, 2.5])
    )

    assert metrics == {
        "train/rigid_loss_object_000": 1.5,
        "train/rigid_loss_object_001": 2.5,
    }


def test_disabled_rigid_loss_returns_zeros_without_evaluating_objective():
    initial_point_clouds = torch.zeros((1, 2, 1, 4, 3))
    prediction = torch.zeros((1, 2, 48, 1, 4, 3))

    def unexpected_objective(*_args, **_kwargs):
        raise AssertionError("disabled rigid loss must not evaluate its objective")

    losses = rigid_loss_terms(
        False,
        initial_point_clouds,
        prediction,
        neighbors=2,
        rigid_loss_fn=unexpected_objective,
    )

    assert torch.equal(losses, torch.zeros((1, 2)))
    assert losses.device == prediction.device
    assert losses.dtype == prediction.dtype


def test_enabled_rigid_loss_uses_the_existing_objective_result():
    initial_point_clouds = torch.zeros((1, 2, 1, 4, 3))
    prediction = torch.zeros((1, 2, 48, 1, 4, 3))
    expected = torch.tensor([[1.5, 2.5]])
    observed = {}

    def objective(initial, predicted, *, neighbors):
        observed.update(initial=initial, predicted=predicted, neighbors=neighbors)
        return expected

    losses = rigid_loss_terms(
        True,
        initial_point_clouds,
        prediction,
        neighbors=3,
        rigid_loss_fn=objective,
    )

    assert losses is expected
    assert observed == {
        "initial": initial_point_clouds,
        "predicted": prediction,
        "neighbors": 3,
    }


def test_disabled_rigid_loss_emits_no_rigid_metrics():
    metrics = add_rigid_metrics({"train/loss": 1.0}, False, torch.tensor([[1.5, 2.5]]))

    assert metrics == {"train/loss": 1.0}


def test_enabled_rigid_loss_emits_sum_and_per_object_metrics():
    metrics = add_rigid_metrics({"train/loss": 1.0}, True, torch.tensor([[1.5, 2.5]]))

    assert metrics == {
        "train/loss": 1.0,
        "train/rigid_loss_sum": 4.0,
        "train/rigid_loss_object_000": 1.5,
        "train/rigid_loss_object_001": 2.5,
    }


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


def test_pc_gradient_norm_uses_pc_gradients_only():
    pc_parameter = torch.nn.Parameter(torch.zeros(2))
    video_parameter = torch.nn.Parameter(torch.zeros(2))
    pc_parameter.grad = torch.tensor([3.0, 4.0])
    video_parameter.grad = torch.tensor([100.0, 100.0])
    model = type(
        "Joint", (), {"pc_model": type("PC", (), {"parameters": lambda self: [pc_parameter]})()}
    )()

    assert pc_gradient_norm(model).item() == 5.0


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
