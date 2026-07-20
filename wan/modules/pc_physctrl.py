"""Dependency-free building blocks for the active PhysCtrl PC architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PhysCtrlAttention(nn.Module):
    """Multi-head attention with CogVideoX's per-head Q/K LayerNorm."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            return projection(tokens).view(
                batch, length, self.heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = (
            split_heads(self.to_q),
            split_heads(self.to_k),
            split_heads(self.to_v),
        )
        output = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


class PhysCtrlLayerNormZero(nn.Module):
    """PhysCtrl's separately gated point/control AdaLN-Zero variant."""

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(
        self,
        points: torch.Tensor,
        controls: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(
            self.act(temb)
        ).chunk(6, dim=-1)
        points = self.norm(points) * (1 + scale[:, None]) + shift[:, None]
        controls = (
            self.norm(controls) * (1 + enc_scale[:, None]) + enc_shift[:, None]
        )
        return points, controls, gate[:, None], enc_gate[:, None]


class PhysCtrlAdaLayerNorm(nn.Module):
    """CogVideoX AdaLayerNorm using a supplied learned timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, values: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(self.act(temb)).chunk(2, dim=-1)
        while shift.ndim < values.ndim:
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        return self.norm(values) * (1 + scale) + shift


class PhysCtrlSpatialTemporalBlock(nn.Module):
    """The active PhysCtrl PC block without generic CogVideoX runtime plumbing."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = PhysCtrlLayerNormZero(dim)
        self.spatial_attention = PhysCtrlAttention(dim, heads)
        self.norm2 = PhysCtrlLayerNormZero(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.temporal_norm = PhysCtrlAdaLayerNorm(dim)
        self.temporal_attention = PhysCtrlAttention(dim, heads)

    def forward(
        self,
        points: torch.Tensor,
        controls: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, count, dim = points.shape
        flat_points = points.reshape(batch * frames, count, dim)
        flat_controls = controls.reshape(batch * frames, 2, dim)
        flat_temb = temb.reshape(batch * frames, dim)

        mod_points, mod_controls, point_gate, control_gate = self.norm1(
            flat_points, flat_controls, flat_temb
        )
        joined = torch.cat((mod_controls, mod_points), dim=1)
        attended = self.spatial_attention(joined)
        flat_controls = flat_controls + control_gate * attended[:, :2]
        flat_points = flat_points + point_gate * attended[:, 2:]

        mod_points, mod_controls, point_gate, control_gate = self.norm2(
            flat_points, flat_controls, flat_temb
        )
        joined = torch.cat((mod_controls, mod_points), dim=1)
        ff_output = self.mlp(joined)
        flat_controls = flat_controls + control_gate * ff_output[:, :2]
        flat_points = flat_points + point_gate * ff_output[:, 2:]

        tracks = flat_points.reshape(batch, frames, count, dim).permute(0, 2, 1, 3)
        tracks = tracks.reshape(batch * count, frames, dim)
        track_temb = temb[:, None].expand(batch, count, frames, dim)
        track_temb = track_temb.reshape(batch * count, frames, dim)
        tracks = tracks + self.temporal_attention(
            self.temporal_norm(tracks, track_temb)
        )
        points = tracks.reshape(batch, count, frames, dim).permute(0, 2, 1, 3)
        controls = flat_controls.reshape(batch, frames, 2, dim)
        return points, controls
