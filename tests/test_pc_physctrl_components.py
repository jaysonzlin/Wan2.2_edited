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


def test_layer_norm_zero_separate_mode_uses_distinct_view_modulation():
    module = PhysCtrlLayerNormZero(4, point_view_gate_mode="separate")
    with torch.no_grad():
        module.linear.weight.zero_()
        module.linear.bias.zero_()
        module.linear.bias[:4].fill_(1.0)
        module.linear.bias[4:8].fill_(2.0)
        module.linear.bias[8:12].fill_(3.0)
        assert module.view_linear is not None
        module.view_linear.weight.zero_()
        module.view_linear.bias.zero_()
        module.view_linear.bias[:4].fill_(4.0)
        module.view_linear.bias[4:8].fill_(5.0)
        module.view_linear.bias[8:12].fill_(6.0)
        module.norm.weight.fill_(1.0)
        module.norm.bias.zero_()
    points = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    views = points.clone()

    point_out, view_out, point_gate, view_gate = module.forward_with_point_views(
        points, views, torch.ones(1, 4)
    )

    expected_point = F.layer_norm(points, (4,)) * 3.0 + 1.0
    expected_view = F.layer_norm(views, (4,)) * 6.0 + 4.0
    torch.testing.assert_close(point_out, expected_point)
    torch.testing.assert_close(view_out, expected_view)
    torch.testing.assert_close(point_gate, torch.full_like(point_gate, 3.0))
    torch.testing.assert_close(view_gate, torch.full_like(view_gate, 6.0))


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


def test_attention_mask_excludes_invalid_key_values():
    attention = PhysCtrlAttention(dim=4, heads=2)
    set_identity_qkvo(attention)
    valid = torch.tensor([[[1.0, 3.0, 2.0, 6.0], [2.0, 4.0, 4.0, 8.0]]])
    baseline = attention(valid)
    joined = torch.cat((valid, torch.full((1, 1, 4), 1e6)), dim=1)

    masked = attention(joined, key_mask=torch.tensor([[True, True, False]]))

    torch.testing.assert_close(masked[:, :2], baseline)


def test_view_tokens_only_enter_spatial_branch_and_invalid_values_are_zeroed():
    block = PhysCtrlSpatialTemporalBlock(dim=4, heads=2)
    points = torch.randn(1, 2, 3, 4)
    views = torch.randn(1, 2, 2, 4)
    mask = torch.tensor([[[True, False], [True, False]]])
    temb = torch.randn(1, 2, 4)
    spatial_shapes, temporal_shapes = [], []
    spatial_hook = block.spatial_attention.register_forward_pre_hook(
        lambda _module, inputs: spatial_shapes.append(inputs[0].shape)
    )
    temporal_hook = block.temporal_attention.register_forward_pre_hook(
        lambda _module, inputs: temporal_shapes.append(inputs[0].shape)
    )
    try:
        output_points, output_views = block.forward_with_point_views(points, views, mask, temb)
    finally:
        spatial_hook.remove()
        temporal_hook.remove()

    assert spatial_shapes == [torch.Size((2, 5, 4))]
    assert temporal_shapes == [torch.Size((3, 2, 4))]
    assert output_points.shape == points.shape
    assert output_views.shape == views.shape
    assert not output_views[:, :, 1].any()


def test_separate_view_gate_mode_adds_view_modulation_to_each_spatial_sublayer():
    block = PhysCtrlSpatialTemporalBlock(
        dim=4, heads=2, point_view_gate_mode="separate"
    )

    assert block.norm1.view_linear is not None
    assert block.norm2.view_linear is not None
    assert "norm1.view_linear.weight" in block.state_dict()
    assert "norm2.view_linear.weight" in block.state_dict()


def test_separate_view_gate_changes_only_view_spatial_residuals():
    block = PhysCtrlSpatialTemporalBlock(
        dim=4, heads=2, point_view_gate_mode="separate"
    )
    set_identity_qkvo(block.spatial_attention)
    with torch.no_grad():
        for norm in (block.norm1, block.norm2):
            norm.linear.weight.zero_()
            norm.linear.bias.zero_()
            assert norm.view_linear is not None
            norm.view_linear.weight.zero_()
            norm.view_linear.bias.zero_()
        block.norm1.view_linear.bias[8:12].fill_(1.0)
        for parameter in block.temporal_attention.parameters():
            parameter.zero_()
    points = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    views = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    temb = torch.ones(1, 1, 4)

    output_points, output_views = block.forward_with_point_views(
        points, views, mask, temb
    )

    torch.testing.assert_close(output_points, points)
    assert not torch.equal(output_views, views)


def test_separate_view_gate_leaves_the_control_forward_path_unchanged():
    torch.manual_seed(9)
    shared_block = PhysCtrlSpatialTemporalBlock(dim=4, heads=2)
    torch.manual_seed(9)
    separate_block = PhysCtrlSpatialTemporalBlock(
        dim=4, heads=2, point_view_gate_mode="separate"
    )
    separate_block.load_state_dict(shared_block.state_dict(), strict=False)
    points = torch.randn(1, 2, 3, 4)
    controls = torch.randn(1, 2, 2, 4)
    temb = torch.randn(1, 2, 4)

    shared_output = shared_block(points, controls, temb)
    separate_output = separate_block(points, controls, temb)

    torch.testing.assert_close(shared_output[0], separate_output[0])
    torch.testing.assert_close(shared_output[1], separate_output[1])
