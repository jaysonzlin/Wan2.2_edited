import torch
import torch.nn.functional as F

from wan.modules.pc_physctrl import (
    PhysCtrlTimestepEmbedding,
    physctrl_position_embedding,
)


def reference_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim / 2)
    angles = positions.reshape(-1, 1).to(torch.float64) / (10000**omega)
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.float32)


def test_position_embedding_uses_physctrl_temporal_spatial_channel_split():
    position = physctrl_position_embedding(num_points=3, num_frames=2, dim=256)

    assert position.shape == (1, 8, 256)
    assert torch.equal(position[:, :2], torch.zeros_like(position[:, :2]))
    expected = torch.cat(
        (
            reference_1d_sincos(torch.arange(2).repeat_interleave(3), 64),
            reference_1d_sincos(torch.arange(3).repeat(2), 192),
        ),
        dim=-1,
    )
    torch.testing.assert_close(position[0, 2:], expected)


def test_timestep_embedding_uses_cogvideox_cos_then_sin_frequencies():
    module = PhysCtrlTimestepEmbedding(8)
    with torch.no_grad():
        module.linear_1.weight.copy_(torch.eye(8))
        module.linear_1.bias.zero_()
        module.linear_2.weight.copy_(torch.eye(8))
        module.linear_2.bias.zero_()

    timesteps = torch.tensor([[0.0, 2.0]])
    half = 4
    frequency = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(half) / half
    )
    raw = torch.cat(
        (
            (timesteps[..., None] * frequency).cos(),
            (timesteps[..., None] * frequency).sin(),
        ),
        dim=-1,
    )

    torch.testing.assert_close(module(timesteps), F.silu(raw), atol=1e-6, rtol=1e-6)
