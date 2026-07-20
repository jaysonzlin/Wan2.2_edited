"""Dependency-free building blocks for the active PhysCtrl PC architecture."""

import math

import torch
import torch.nn as nn


def physctrl_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Return PhysCtrl's fixed sine/cosine embedding for scalar positions."""
    if dim % 2:
        raise ValueError("dim must be even")
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float64)
    omega = 1.0 / 10000 ** (omega / (dim / 2.0))
    angles = positions.reshape(-1, 1).to(torch.float64) * omega
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.float32)


def physctrl_position_embedding(
    num_points: int, num_frames: int, dim: int
) -> torch.Tensor:
    """Build the zero-control, temporal-plus-point PhysCtrl token position table."""
    if dim % 4:
        raise ValueError("dim must be divisible by 4")
    temporal = physctrl_1d_sincos(
        torch.arange(num_frames).repeat_interleave(num_points), dim // 4
    )
    spatial = physctrl_1d_sincos(
        torch.arange(num_points).repeat(num_frames), 3 * dim // 4
    )
    points = torch.cat((temporal, spatial), dim=-1)
    controls = torch.zeros(1, 2, dim, dtype=points.dtype)
    return torch.cat((controls, points.unsqueeze(0)), dim=1)


class PhysCtrlTimestepEmbedding(nn.Module):
    """CogVideoX timestep Fourier features followed by its learned MLP."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        angles = timesteps.float()[..., None] * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.linear_2(self.act(self.linear_1(embedding)))
