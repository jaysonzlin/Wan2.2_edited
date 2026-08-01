"""Aligned DDPM batches and per-object objectives for joint Wan--PhysCtrl training."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultiObjectPCDDPMBatch:
    """Noised future trajectories for every object in a batch-size-one video clip."""

    model_input: torch.Tensor
    target: torch.Tensor
    frame_times: torch.Tensor
    timesteps: torch.Tensor


def make_aligned_multi_object_pc_ddpm_batch(
    point_clouds: torch.Tensor, scheduler, generator: torch.Generator
) -> MultiObjectPCDDPMBatch:
    """Noise frames 1--48 with one DDPM timestep per video and independent object noise.

    ``point_clouds`` has shape ``[batch, objects, 49, 1, points, xyz]``. The same sampled
    discrete timestep is expanded to all objects from the same video, whereas the Gaussian noise
    remains independently sampled at every element.
    """
    if point_clouds.ndim != 6 or point_clouds.shape[2] != 49:
        raise ValueError("point_clouds must have shape [B, K, 49, 1, N, 3]")
    batch_size, object_count = point_clouds.shape[:2]
    future_points = point_clouds[:, :, 1:]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=point_clouds.device,
        generator=generator,
    )
    noise = torch.randn(
        future_points.shape,
        device=future_points.device,
        dtype=future_points.dtype,
        generator=generator,
    )
    flat_future = future_points.flatten(0, 1)
    flat_noise = noise.flatten(0, 1)
    flat_timesteps = timesteps[:, None].expand(batch_size, object_count).reshape(-1)
    noised = scheduler.add_noise(flat_future, flat_noise, flat_timesteps).unflatten(
        0, (batch_size, object_count)
    )
    frame_times = timesteps[:, None, None].expand(batch_size, object_count, 49).to(
        point_clouds.dtype
    )
    return MultiObjectPCDDPMBatch(
        model_input=noised,
        target=future_points,
        frame_times=frame_times,
        timesteps=timesteps,
    )


def per_object_pc_x0_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return individual object x0 losses and their unaveraged total."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim < 3:
        raise ValueError("prediction must include batch and object dimensions")
    losses = (prediction - target).square().flatten(start_dim=2).mean(dim=2)
    return losses, losses.sum()
