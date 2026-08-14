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
    known_frames: int = 1,
) -> PCFlowBatch:
    """Create shifted flow-matching data for future point displacements."""
    if (
        future_points.ndim != 5
        or future_points.shape[2] != 1
        or future_points.shape[-1] != 3
    ):
        raise ValueError("future_points must have shape (B, F, 1, N, 3)")
    if init_pc.shape == (
        future_points.shape[0],
        known_frames,
        1,
        future_points.shape[3],
        3,
    ):
        source_points = init_pc[:, :1]
    elif known_frames == 1 and init_pc.shape == (
        future_points.shape[0],
        1,
        future_points.shape[3],
        3,
    ):
        source_points = init_pc.unsqueeze(1)
    else:
        raise ValueError("init_pc must have shape (B, K, 1, N, 3)")
    if time_shift <= 0 or num_train_timesteps <= 0:
        raise ValueError("time_shift and num_train_timesteps must be positive")

    displacements = future_points - source_points
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
        (
            torch.zeros_like(times[:, None]).expand(-1, known_frames),
            times[:, None].expand(-1, future_points.shape[1]),
        ),
        dim=1,
    ).mul(num_train_timesteps)
    return PCFlowBatch(model_input, noise - displacements, frame_times)


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a shape-safe objective mean-squared loss."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return F.mse_loss(prediction, target)
