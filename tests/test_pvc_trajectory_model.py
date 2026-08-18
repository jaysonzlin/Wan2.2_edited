import pytest
import torch

from wan.modules.pvc_trajectory import PVCTrajectoryModel


def make_model(feature_dim=5):
    return PVCTrajectoryModel(
        n_points=8, n_future_frames=45, latent_dim=64, n_layers=1,
        num_heads=1, utonia_feature_dim=feature_dim,
    )


def pvc_inputs(batch=2, feature_dim=5):
    return (
        torch.randn(batch, 45, 1, 8, 3),
        torch.tensor([[0.0] * 4 + [500.0] * 45] * batch),
        torch.randn(batch, 4, 1, 8, 3),
        torch.randn(batch, 49, 8, 3),
        torch.tensor([[True] * 3 + [False] * 5]).expand(batch, 49, -1).clone(),
        torch.randn(batch, 8, feature_dim),
        torch.randn(batch, 49, 8, feature_dim),
    )


def test_pvc_model_predicts_only_45_trajectory_frames():
    model = make_model()

    output = model(*pvc_inputs())

    assert output.shape == (2, 45, 1, 8, 3)


def test_pvc_model_rejects_invalid_view_mask_shape_or_dtype():
    model = make_model()
    inputs = list(pvc_inputs())
    inputs[4] = torch.ones((2, 49, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="point_view_mask"):
        model(*inputs)


def test_pvc_separate_view_gate_mode_constructs_independent_view_modulators():
    model = PVCTrajectoryModel(
        n_points=8,
        n_future_frames=45,
        latent_dim=64,
        n_layers=1,
        num_heads=1,
        utonia_feature_dim=5,
        point_view_gate_mode="separate",
    )

    state = model.state_dict()

    assert "blocks.0.norm1.view_linear.weight" in state
    assert "blocks.0.norm2.view_linear.weight" in state


def test_pvc_shared_mode_preserves_the_legacy_view_block_layout_and_behavior():
    torch.manual_seed(7)
    default_model = make_model()
    torch.manual_seed(7)
    shared_model = PVCTrajectoryModel(
        n_points=8,
        n_future_frames=45,
        latent_dim=64,
        n_layers=1,
        num_heads=1,
        utonia_feature_dim=5,
        point_view_gate_mode="shared",
    )
    inputs = pvc_inputs(batch=1)

    expected_block_shapes = {
        "norm1.linear.weight": (384, 64),
        "norm1.linear.bias": (384,),
        "norm2.linear.weight": (384, 64),
        "norm2.linear.bias": (384,),
    }
    assert {
        key: tuple(value.shape)
        for key, value in default_model.blocks[0].state_dict().items()
        if key in expected_block_shapes
    } == expected_block_shapes
    assert not any("view_linear" in key for key in default_model.state_dict())
    assert default_model.state_dict().keys() == shared_model.state_dict().keys()
    for key, value in default_model.state_dict().items():
        torch.testing.assert_close(value, shared_model.state_dict()[key])
    torch.testing.assert_close(default_model(*inputs), shared_model(*inputs))

    block = default_model.blocks[0]
    points = torch.randn(1, 2, 3, 64)
    views = torch.randn(1, 2, 4, 64)
    mask = torch.tensor([[[True, True, False, False], [True, False, True, False]]])
    temb = torch.randn(1, 2, 64)
    batch, frames, count, dim = points.shape
    flat_points = points.reshape(batch * frames, count, dim)
    flat_views = views.reshape(batch * frames, views.shape[2], dim)
    flat_mask = mask.reshape(batch * frames, views.shape[2])
    flat_temb = temb.reshape(batch * frames, dim)
    joined = torch.cat((flat_points, flat_views), dim=1)
    key_mask = torch.cat(
        (
            torch.ones(batch * frames, count, dtype=torch.bool),
            flat_mask,
        ),
        dim=1,
    )
    mod_joined, _, gate, _ = block.norm1(joined, None, flat_temb)
    joined = joined + gate * block.spatial_attention(mod_joined, key_mask=key_mask)
    mod_joined, _, gate, _ = block.norm2(joined, None, flat_temb)
    joined = joined + gate * block.mlp(mod_joined)
    legacy_points, legacy_views = joined.split((count, views.shape[2]), dim=1)
    legacy_views = legacy_views.masked_fill(~flat_mask[..., None], 0)
    tracks = legacy_points.reshape(batch, frames, count, dim).permute(0, 2, 1, 3)
    tracks = tracks.reshape(batch * count, frames, dim)
    track_temb = temb[:, None].expand(batch, count, frames, dim)
    track_temb = track_temb.reshape(batch * count, frames, dim)
    tracks = tracks + block.temporal_attention(block.temporal_norm(tracks, track_temb))
    legacy_points = tracks.reshape(batch, count, frames, dim).permute(0, 2, 1, 3)
    legacy_views = legacy_views.reshape(batch, frames, views.shape[2], dim)

    output_points, output_views = block.forward_with_point_views(
        points, views, mask, temb
    )

    torch.testing.assert_close(output_points, legacy_points)
    torch.testing.assert_close(output_views, legacy_views)
