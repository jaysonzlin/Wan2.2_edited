import torch
import torch.nn.functional as F

from wan.modules.pc_physctrl import (
    PhysCtrlAdaLayerNorm,
    PhysCtrlAttention,
    PhysCtrlLayerNormZero,
    PhysCtrlSpatialTemporalBlock,
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


def set_identity_qkvo(attention: torch.nn.Module) -> None:
    with torch.no_grad():
        for projection in (
            attention.to_q,
            attention.to_k,
            attention.to_v,
            attention.to_out,
        ):
            projection.weight.copy_(torch.eye(projection.in_features))
            projection.bias.zero_()
        attention.q_norm.weight.fill_(1)
        attention.q_norm.bias.zero_()
        attention.k_norm.weight.fill_(1)
        attention.k_norm.bias.zero_()


def test_attention_normalizes_q_and_k_per_head_before_sdpa():
    attention = PhysCtrlAttention(dim=4, heads=2)
    set_identity_qkvo(attention)
    tokens = torch.tensor([[[1.0, 3.0, 2.0, 6.0], [2.0, 4.0, 4.0, 8.0]]])
    heads = tokens.reshape(1, 2, 2, 2).transpose(1, 2)
    q = F.layer_norm(
        heads, (2,), attention.q_norm.weight, attention.q_norm.bias, 1e-6
    )
    k = F.layer_norm(
        heads, (2,), attention.k_norm.weight, attention.k_norm.bias, 1e-6
    )
    expected = F.scaled_dot_product_attention(q, k, heads)
    expected = expected.transpose(1, 2).reshape(1, 2, 4)

    torch.testing.assert_close(attention(tokens), expected)


def test_layer_norm_zero_uses_distinct_point_and_control_modulation():
    module = PhysCtrlLayerNormZero(4)
    with torch.no_grad():
        module.linear.weight.zero_()
        module.linear.bias.zero_()
        module.linear.bias[:4].fill_(1.0)
        module.linear.bias[8:12].fill_(3.0)
        module.linear.bias[12:16].fill_(2.0)
        module.linear.bias[20:24].fill_(4.0)
        module.norm.weight.fill_(1.0)
        module.norm.bias.zero_()
    points = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    controls = points.expand(-1, 2, -1)

    point_out, control_out, point_gate, control_gate = module(
        points, controls, torch.ones(1, 4)
    )

    torch.testing.assert_close(control_out[:, :1] - point_out, torch.ones_like(point_out))
    torch.testing.assert_close(point_gate, torch.full_like(point_gate, 3.0))
    torch.testing.assert_close(control_gate, torch.full_like(control_gate, 4.0))


def test_adaptive_layer_norm_accepts_per_frame_timestep_embeddings():
    module = PhysCtrlAdaLayerNorm(4)
    with torch.no_grad():
        module.linear.weight.zero_()
        module.linear.bias.zero_()
        module.linear.bias[:4].fill_(2.0)
        module.linear.bias[4:].fill_(3.0)
        module.norm.weight.fill_(1.0)
        module.norm.bias.zero_()
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]])
    temb = torch.zeros(1, 2, 4)
    expected = F.layer_norm(values, (4,), module.norm.weight, module.norm.bias, 1e-5)
    expected = expected * 4.0 + 2.0

    torch.testing.assert_close(module(values, temb), expected)


def test_block_applies_spatial_attention_to_two_controls_plus_points():
    block = PhysCtrlSpatialTemporalBlock(dim=4, heads=2)
    seen = []
    handle = block.spatial_attention.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].shape)
    )
    try:
        block(
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 2, 4),
            torch.randn(2, 3, 4),
        )
    finally:
        handle.remove()

    assert seen == [torch.Size((6, 7, 4))]


def test_zero_gates_and_zero_temporal_attention_preserve_both_streams():
    block = PhysCtrlSpatialTemporalBlock(dim=4, heads=2)
    with torch.no_grad():
        for parameter in block.norm1.linear.parameters():
            parameter.zero_()
        for parameter in block.norm2.linear.parameters():
            parameter.zero_()
        for parameter in block.temporal_attention.parameters():
            parameter.zero_()
    points = torch.randn(1, 2, 3, 4)
    controls = torch.randn(1, 2, 2, 4)

    output_points, output_controls = block(points, controls, torch.randn(1, 2, 4))

    torch.testing.assert_close(output_points, points)
    torch.testing.assert_close(output_controls, controls)
