"""DDPM x0 batches for fixed point-cloud trajectories."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PCDDPMBatch:
    model_input: torch.Tensor
    target: torch.Tensor
    frame_times: torch.Tensor
    timesteps: torch.Tensor


def make_pc_ddpm_batch(
    future_points, scheduler, generator, known_frames: int = 1
) -> PCDDPMBatch:
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (future_points.shape[0],),
        device=future_points.device,
        generator=generator,
    )
    noise = torch.randn(
        future_points.shape,
        device=future_points.device,
        dtype=future_points.dtype,
        generator=generator,
    )
    future_times = timesteps[:, None].expand(-1, future_points.shape[1]).to(
        future_points.dtype
    )
    frame_times = (
        torch.cat((torch.zeros_like(future_times[:, :known_frames]), future_times), dim=1)
        if known_frames > 1
        else timesteps[:, None].expand(-1, known_frames + future_points.shape[1]).to(
            future_points.dtype
        )
    )
    return PCDDPMBatch(
        scheduler.add_noise(future_points, noise, timesteps),
        future_points,
        frame_times,
        timesteps,
    )
