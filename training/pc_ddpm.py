"""DDPM x0 batches for fixed point-cloud trajectories."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PCDDPMBatch:
    model_input: torch.Tensor
    target: torch.Tensor
    frame_times: torch.Tensor
    timesteps: torch.Tensor


def make_pc_ddpm_batch(future_points, scheduler, generator) -> PCDDPMBatch:
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
    return PCDDPMBatch(
        scheduler.add_noise(future_points, noise, timesteps),
        future_points,
        timesteps[:, None].expand(-1, 49).to(future_points.dtype),
        timesteps,
    )
