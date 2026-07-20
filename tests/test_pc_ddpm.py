import torch

from training.pc_ddpm import make_pc_ddpm_batch


class FakeDDPMScheduler:
    config = type("Config", (), {"num_train_timesteps": 1000})()

    def add_noise(self, sample, noise, timesteps):
        return sample + noise


def test_ddpm_batch_noises_absolute_positions_and_repeats_time():
    future = torch.full((2, 48, 1, 2, 3), 7.0)
    batch = make_pc_ddpm_batch(future, FakeDDPMScheduler(), torch.Generator().manual_seed(0))

    assert batch.target is future
    assert batch.model_input.shape == future.shape
    assert batch.timesteps.dtype == torch.long
    assert torch.equal(batch.frame_times, batch.timesteps[:, None].expand(-1, 49).to(future.dtype))
