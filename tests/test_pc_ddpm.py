import pytest
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


def test_ddpm_batch_uses_zero_times_for_four_clean_history_frames():
    future = torch.full((2, 45, 1, 2, 3), 7.0)
    batch = make_pc_ddpm_batch(
        future,
        FakeDDPMScheduler(),
        torch.Generator().manual_seed(0),
        known_frames=4,
    )

    assert batch.target is future
    assert batch.model_input.shape == future.shape
    assert torch.equal(batch.frame_times[:, :4], torch.zeros(2, 4))
    assert torch.equal(
        batch.frame_times[:, 4:],
        batch.timesteps[:, None].expand(-1, 45).to(future.dtype),
    )


@pytest.mark.parametrize(
    "future_frames, known_frames",
    [(48, 4), (45, 1), (45, 2), (45, 5), (48, True)],
)
def test_ddpm_batch_rejects_unsupported_temporal_layouts(
    future_frames, known_frames
):
    future = torch.zeros(1, future_frames, 1, 2, 3)

    with pytest.raises(ValueError):
        make_pc_ddpm_batch(
            future,
            FakeDDPMScheduler(),
            torch.Generator().manual_seed(0),
            known_frames=known_frames,
        )
