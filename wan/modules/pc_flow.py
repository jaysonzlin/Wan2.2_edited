"""Factorized Wan-style flow transformer for fixed point-cloud trajectories."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WanRMSNorm


def _sinusoidal(times: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=times.device, dtype=times.dtype) / half)
    values = times[..., None] * freqs
    return torch.cat((values.sin(), values.cos()), dim=-1)


class PointEmbed(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 96):
        super().__init__()
        basis = torch.pow(2, torch.arange(hidden_dim // 6)).float() * math.pi
        basis = torch.stack((
            torch.cat((basis, torch.zeros_like(basis), torch.zeros_like(basis))),
            torch.cat((torch.zeros_like(basis), basis, torch.zeros_like(basis))),
            torch.cat((torch.zeros_like(basis), torch.zeros_like(basis), basis)),
        ))
        self.register_buffer("basis", basis)
        self.projection = nn.Linear(hidden_dim + 3, dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bnd,de->bne", points, self.basis)
        return self.projection(torch.cat((projections.sin(), projections.cos(), points), dim=-1))


class PCSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("latent_dim must divide evenly into num_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q, self.norm_k = WanRMSNorm(dim), WanRMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        q = self.norm_q(self.q(x)).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(self.k(x)).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, -1))


class PCAdaptiveModulation(nn.Module):
    def __init__(self, dim: int, chunks: int = 6):
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, chunks * dim))

    def forward(self, time_embedding: torch.Tensor):
        return self.projection(time_embedding).chunk(6, dim=-1)


class PCSpatialTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.spatial_norm, self.temporal_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.spatial_attention, self.temporal_attention = PCSelfAttention(dim, heads), PCSelfAttention(dim, heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.modulation = PCAdaptiveModulation(dim)

    def forward(self, points: torch.Tensor, conditions: torch.Tensor, times: torch.Tensor):
        b, frames, count, dim = points.shape
        time = _sinusoidal(times, dim)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time)
        joint = torch.cat((conditions, points), dim=2).reshape(b * frames, count + 2, dim)
        sa, ss, sg = (value.reshape(b * frames, 1, dim) for value in (shift_a, scale_a, gate_a))
        sm, ms, mg = (value.reshape(b * frames, 1, dim) for value in (shift_m, scale_m, gate_m))
        joint = joint + sg * self.spatial_attention(self.spatial_norm(joint) * (1 + ss) + sa)
        joint = joint + mg * self.mlp(self.mlp_norm(joint) * (1 + ms) + sm)
        joint = joint.reshape(b, frames, count + 2, dim)
        conditions, points = joint[:, :, :2], joint[:, :, 2:]
        tracks = points.permute(0, 2, 1, 3).reshape(b * count, frames, dim)
        temporal = time[:, None].expand(b, count, frames, dim).reshape(b * count, frames, dim)
        tracks = tracks + self.temporal_attention(self.temporal_norm(tracks) + temporal)
        return tracks.reshape(b, count, frames, dim).permute(0, 2, 1, 3), conditions


class PCFlowHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.projection = nn.Linear(dim, 3)

    def forward(self, points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(_sinusoidal(times, points.shape[-1])).chunk(2, dim=-1)
        return self.projection(self.norm(points) * (1 + scale[:, :, None]) + shift[:, :, None])


class PCFlowModel(nn.Module):
    def __init__(self, n_points=2048, n_future_frames=48, latent_dim=256, n_layers=8, num_heads=4, point_embed=True):
        super().__init__()
        self.n_points, self.n_future_frames, self.latent_dim = n_points, n_future_frames, latent_dim
        self.input_encoder = PointEmbed(latent_dim) if point_embed else nn.Linear(3, latent_dim)
        self.linear_velocity_encoder, self.angular_velocity_encoder = nn.Linear(3, latent_dim), nn.Linear(3, latent_dim)
        self.blocks = nn.ModuleList(PCSpatialTemporalBlock(latent_dim, num_heads) for _ in range(n_layers))
        self.flow_head = PCFlowHead(latent_dim)
        self.register_buffer("point_position", _sinusoidal(torch.arange(n_points).float(), latent_dim), persistent=False)
        self.register_buffer("frame_position", _sinusoidal(torch.arange(n_future_frames + 1).float(), latent_dim), persistent=False)

    def forward(self, noisy_displacements, frame_times, init_pc, initial_linear_velocity, initial_angular_velocity):
        b = noisy_displacements.shape[0]
        expected = (b, self.n_future_frames, 1, self.n_points, 3)
        if noisy_displacements.shape != expected:
            raise ValueError(f"noisy_displacements must have shape {expected}")
        if init_pc.shape != (b, 1, self.n_points, 3):
            raise ValueError("init_pc must have shape (B, 1, N, 3)")
        if frame_times.shape != (b, self.n_future_frames + 1):
            raise ValueError("frame_times must have shape (B, 49)")
        if not torch.equal(frame_times[:, 0], torch.zeros_like(frame_times[:, 0])):
            raise ValueError("frame_times[:, 0] must be zero")
        if initial_linear_velocity.shape != (b, 1, 3) or initial_angular_velocity.shape != (b, 1, 3):
            raise ValueError("initial velocities must have shape (B, 1, 3)")
        future_positions = init_pc.unsqueeze(1) + noisy_displacements
        coordinates = torch.cat((init_pc.unsqueeze(1), future_positions), dim=1).squeeze(2)
        points = self.input_encoder(coordinates.reshape(-1, self.n_points, 3)).reshape(b, self.n_future_frames + 1, self.n_points, self.latent_dim)
        points = points + self.point_position[None, None] + self.frame_position[None, :, None]
        conditions = torch.stack((self.linear_velocity_encoder(initial_linear_velocity.squeeze(1)), self.angular_velocity_encoder(initial_angular_velocity.squeeze(1))), dim=1)
        conditions = conditions[:, None].expand(-1, self.n_future_frames + 1, -1, -1)
        for block in self.blocks:
            points, conditions = block(points, conditions, frame_times)
        return self.flow_head(points[:, 1:], frame_times[:, 1:]).unsqueeze(2)
