"""Flow-matching batches and loss for point-cloud trajectories."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PCFlowBatch:
    model_input: torch.Tensor
    velocity_target: torch.Tensor
    frame_times: torch.Tensor


def make_pc_flow_batch(
    future_points: torch.Tensor,
    init_pc: torch.Tensor,
    generator: torch.Generator,
    time_shift: float,
    num_train_timesteps: int,
) -> PCFlowBatch:
    """Create shifted flow-matching data for future point displacements."""
    if future_points.ndim != 5 or future_points.shape[1:3] != (48, 1) or future_points.shape[-1] != 3:
        raise ValueError("future_points must have shape (B, 48, 1, N, 3)")
    if init_pc.shape != (future_points.shape[0], 1, future_points.shape[3], 3):
        raise ValueError("init_pc must have shape (B, 1, N, 3)")
    if time_shift <= 0 or num_train_timesteps <= 0:
        raise ValueError("time_shift and num_train_timesteps must be positive")

    displacements = future_points - init_pc.unsqueeze(1)
    uniform_times = torch.rand(
        (future_points.shape[0],),
        device=future_points.device,
        dtype=future_points.dtype,
        generator=generator,
    )
    times = time_shift * uniform_times / (1 + (time_shift - 1) * uniform_times)
    noise = torch.randn(
        displacements.shape,
        device=displacements.device,
        dtype=displacements.dtype,
        generator=generator,
    )
    interpolation = times[:, None, None, None, None]
    model_input = (1 - interpolation) * displacements + interpolation * noise
    frame_times = torch.cat(
        (torch.zeros_like(times[:, None]), times[:, None].expand(-1, 48)), dim=1
    ).mul(num_train_timesteps)
    return PCFlowBatch(model_input, noise - displacements, frame_times)


def flow_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a shape-safe mean-squared flow loss."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return F.mse_loss(prediction, target)
