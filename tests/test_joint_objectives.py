import torch

from training.joint_objectives import (
    make_aligned_multi_object_pc_ddpm_batch,
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
