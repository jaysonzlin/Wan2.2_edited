import pytest
import torch

from training.joint_objectives import (
    make_aligned_multi_object_pc_ddpm_batch,
    per_object_baseline_corrected_deform_loss,
    per_object_rigid_edge_length_loss,
    per_object_pc_x0_mse,
)


class RecordingDDPMScheduler:
    config = type("Config", (), {"num_train_timesteps": 1000})()

    def __init__(self):
        self.noise = None
        self.timesteps = None

    def add_noise(self, sample, noise, timesteps):
        self.noise = noise
        self.timesteps = timesteps
        return sample + noise


def test_multi_object_ddpm_aligns_timestep_but_uses_independent_noise():
    points = torch.full((1, 3, 49, 1, 2, 3), 7.0)
    scheduler = RecordingDDPMScheduler()

    batch = make_aligned_multi_object_pc_ddpm_batch(
        points, scheduler, torch.Generator().manual_seed(42)
    )

    assert torch.equal(batch.target, points[:, :, 1:])
    assert batch.model_input.shape == (1, 3, 48, 1, 2, 3)
    assert batch.timesteps.shape == (1,)
    assert torch.equal(
        scheduler.timesteps,
        batch.timesteps[:, None].expand(1, 3).reshape(-1),
    )
    assert torch.equal(
        batch.frame_times,
        batch.timesteps[:, None, None].expand(1, 3, 49).to(points.dtype),
    )
    assert not torch.equal(scheduler.noise[0], scheduler.noise[1])


def test_per_object_pc_x0_mse_returns_each_loss_and_their_sum():
    target = torch.zeros((1, 2, 2, 1, 1, 3))
    prediction = target.clone()
    prediction[:, 0].fill_(2.0)
    prediction[:, 1].fill_(3.0)

    losses, total = per_object_pc_x0_mse(prediction, target)

    assert torch.equal(losses, torch.tensor([[4.0, 9.0]]))
    assert total.item() == 13.0


def _two_object_initial_point_clouds() -> torch.Tensor:
    first_object = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ).reshape(1, 1, 1, 4, 3)
    second_object = first_object + torch.tensor([[[[3.0, -2.0, 1.0]]]])
    return torch.cat((first_object, second_object), dim=1)


def test_per_object_rigid_edge_length_loss_is_zero_for_rigid_motion():
    initial = _two_object_initial_point_clouds()
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([[[[2.0, 4.0, -3.0]], [[-1.0, 3.0, 2.0]]]])
    rigid_future = initial[:, :, 0] @ rotation.T + translation
    prediction = (
        rigid_future.unsqueeze(2).unsqueeze(3).expand(-1, -1, 48, 1, -1, -1).clone()
    )

    losses = per_object_rigid_edge_length_loss(initial, prediction, neighbors=2)

    assert losses.shape == (1, 2)
    assert torch.allclose(losses, torch.zeros_like(losses), atol=1e-6)


def test_per_object_rigid_edge_length_loss_is_positive_for_nonrigid_motion():
    initial = _two_object_initial_point_clouds()
    prediction = (
        initial[:, :, 0].unsqueeze(2).unsqueeze(3).expand(-1, -1, 48, 1, -1, -1).clone()
    )
    prediction[:, 1, :, :, 0] *= 2

    losses = per_object_rigid_edge_length_loss(initial, prediction, neighbors=2)

    assert losses[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert losses[0, 1].item() > 0


def test_per_object_rigid_edge_length_loss_rejects_invalid_neighbor_count():
    initial = torch.zeros((1, 1, 1, 4, 3))
    prediction = torch.zeros((1, 1, 48, 1, 4, 3))

    with pytest.raises(ValueError, match="neighbors"):
        per_object_rigid_edge_length_loss(initial, prediction, neighbors=4)


def _static_deformation_inputs():
    points = torch.tensor(
        [
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 5.0],
            [3.0, 5.0, 3.0],
            [3.0, 5.0, 5.0],
            [5.0, 3.0, 3.0],
            [5.0, 3.0, 5.0],
            [5.0, 5.0, 3.0],
            [5.0, 5.0, 5.0],
        ]
    )
    initial = points.reshape(1, 1, 1, 8, 3)
    future = points.reshape(1, 1, 1, 1, 8, 3).expand(-1, -1, 48, -1, -1, -1).clone()
    identity = torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3)
    deformation = identity.expand(1, 1, 49, 1, 8, -1, -1).clone()
    affine_velocity = torch.zeros_like(deformation)
    return {
        "initial": initial,
        "future": future,
        "deformation": deformation,
        "affine_velocity": affine_velocity,
        "volume": torch.ones((1, 1, 1, 8)),
        "baseline": torch.zeros((1, 1, 47, 1, 8, 3, 3)),
        "origin": torch.zeros((1, 1, 3)),
        "scale": torch.ones((1, 1, 1)),
    }


def test_deformation_loss_cancels_saved_gt_baseline():
    fields = _static_deformation_inputs()

    loss = per_object_baseline_corrected_deform_loss(
        fields["initial"],
        fields["future"],
        fields["deformation"],
        fields["affine_velocity"],
        fields["volume"],
        fields["baseline"],
        fields["origin"],
        fields["scale"],
        grid_size=8,
    )

    assert loss.shape == (1, 1)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_deformation_loss_increases_for_perturbed_future_and_preserves_gradients():
    fields = _static_deformation_inputs()
    prediction = fields["future"].clone().requires_grad_()
    prediction.data[:, :, 10, :, 0, 0] += 0.1

    loss = per_object_baseline_corrected_deform_loss(
        fields["initial"],
        prediction,
        fields["deformation"],
        fields["affine_velocity"],
        fields["volume"],
        fields["baseline"],
        fields["origin"],
        fields["scale"],
        grid_size=8,
    )
    loss.sum().backward()

    assert loss.item() > 0
    assert prediction.grad is not None
    assert prediction.grad.abs().sum().item() > 0


def test_deformation_loss_handles_an_outside_predicted_position():
    """An untrained x0 trajectory must not index outside the fixed P2G grid."""
    fields = _static_deformation_inputs()
    prediction = fields["future"].clone()
    prediction[:, :, 0, :, 0].fill_(1_000.0)

    loss = per_object_baseline_corrected_deform_loss(
        fields["initial"],
        prediction,
        fields["deformation"],
        fields["affine_velocity"],
        fields["volume"],
        fields["baseline"],
        fields["origin"],
        fields["scale"],
        grid_size=8,
    )

    assert torch.isfinite(loss).all()
