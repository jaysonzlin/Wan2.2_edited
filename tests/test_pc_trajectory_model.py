import pytest
import torch

from wan.modules.pc_trajectory import PCTrajectoryModel


def make_tiny_model(
    objective_type="flow",
    utonia_feature_dim=None,
    conditioning="velocity",
    history_frames=1,
):
    conditioning_kwargs = {}
    if conditioning != "velocity" or history_frames != 1:
        conditioning_kwargs = {
            "conditioning": conditioning,
            "history_frames": history_frames,
        }
    return PCTrajectoryModel(
        n_points=8,
        n_future_frames=48,
        latent_dim=64,
        n_layers=1,
        num_heads=1,
        point_embed=True,
        objective_type=objective_type,
        utonia_feature_dim=utonia_feature_dim,
        **conditioning_kwargs,
    )


def test_model_returns_direct_future_flow_shape():
    model = make_tiny_model()

    output = model(
        torch.randn(2, 48, 1, 8, 3),
        torch.tensor([[0.0] + [500.0] * 48] * 2),
        torch.randn(2, 1, 8, 3),
        torch.randn(2, 1, 3),
        torch.randn(2, 1, 3),
    )

    assert output.shape == (2, 48, 1, 8, 3)


def test_model_rejects_nonzero_source_time():
    model = make_tiny_model()

    with pytest.raises(ValueError, match=r"frame_times\[:, 0\] must be zero"):
        model(
            torch.zeros(1, 48, 1, 8, 3),
            torch.ones(1, 49),
            torch.zeros(1, 1, 8, 3),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        )


def test_model_embeds_future_flow_states_as_absolute_positions():
    model = make_tiny_model()
    captured = {}

    def capture_coordinates(_module, inputs):
        captured["coordinates"] = inputs[0].detach().clone()

    handle = model.input_encoder.register_forward_pre_hook(capture_coordinates)
    initial = torch.full((1, 1, 8, 3), 10.0)
    flow_state = torch.full((1, 48, 1, 8, 3), 2.0)
    try:
        model(
            flow_state,
            torch.tensor([[0.0] + [500.0] * 48]),
            initial,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        )
    finally:
        handle.remove()

    embedded_frames = captured["coordinates"].reshape(1, 49, 8, 3)
    assert torch.equal(embedded_frames[:, :1], initial)
    assert torch.equal(
        embedded_frames[:, 1:], torch.full_like(flow_state.squeeze(2), 12.0)
    )


def test_zero_output_head_never_adds_source_coordinates():
    model = make_tiny_model()
    torch.nn.init.zeros_(model.output_head.projection.weight)
    torch.nn.init.zeros_(model.output_head.projection.bias)

    output = model(
        torch.zeros(1, 48, 1, 8, 3),
        torch.tensor([[0.0] + [1.0] * 48]),
        torch.full((1, 1, 8, 3), 9.0),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
    )

    assert torch.equal(output, torch.zeros_like(output))


def test_ddpm_model_adds_source_to_zero_predicted_offset():
    model = make_tiny_model(objective_type="ddpm")
    torch.nn.init.zeros_(model.output_head.projection.weight)
    torch.nn.init.zeros_(model.output_head.projection.bias)
    source = torch.full((1, 1, 8, 3), 9.0)

    output = model(
        torch.zeros(1, 48, 1, 8, 3),
        torch.full((1, 49), 500.0),
        source,
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
    )

    assert torch.equal(output, source.unsqueeze(1).expand_as(output))


def test_default_ddpm_reuses_one_timestep_embedding_at_all_49_frames():
    model = make_tiny_model(objective_type="ddpm")
    captured = {}
    handle = model.time_embedding.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("temb", output.detach().clone())
    )
    try:
        model(
            torch.zeros(1, 48, 1, 8, 3),
            torch.full((1, 49), 123.0),
            torch.zeros(1, 1, 8, 3),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        )
    finally:
        handle.remove()

    assert captured["temb"].shape == (1, 49, 64)
    assert torch.equal(
        captured["temb"][:, :1].expand_as(captured["temb"]), captured["temb"]
    )


def test_parity_backbone_has_no_dropout_or_wan_rms_norm_modules():
    model = make_tiny_model(objective_type="ddpm")

    names = {type(module).__name__ for module in model.modules()}

    assert "Dropout" not in names
    assert "WanRMSNorm" not in names


def test_state_helpers_preserve_legacy_forward_result():
    model = make_tiny_model(objective_type="ddpm")
    noisy = torch.randn(1, 48, 1, 8, 3)
    frame_times = torch.full((1, 49), 123.0)
    initial = torch.randn(1, 1, 8, 3)
    linear = torch.randn(1, 1, 3)
    angular = torch.randn(1, 1, 3)

    expected = model(noisy, frame_times, initial, linear, angular)
    points, controls, temb = model.encode_states(
        noisy, frame_times, initial, linear, angular
    )
    for block in model.blocks:
        points, controls = block(points, controls, temb)
    actual = model.decode_states(points, temb, initial)

    torch.testing.assert_close(actual, expected)


def test_model_rejects_non_physctrl_head_width_or_point_encoder():
    with pytest.raises(ValueError, match="num_heads must equal latent_dim // 64"):
        PCTrajectoryModel(
            n_points=8,
            n_future_frames=48,
            latent_dim=64,
            n_layers=1,
            num_heads=2,
        )
    with pytest.raises(ValueError, match="point_embed must be true"):
        PCTrajectoryModel(
            n_points=8,
            n_future_frames=48,
            latent_dim=64,
            n_layers=1,
            num_heads=1,
            point_embed=False,
        )


def test_model_fuses_per_point_utonia_features_across_all_frames():
    model = make_tiny_model(utonia_feature_dim=5)
    captured = {}
    handle = model.utonia_feature_projection.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("tokens", inputs[0].detach().clone())
    )
    features = torch.randn(2, 8, 5)
    try:
        output = model(
            torch.randn(2, 48, 1, 8, 3),
            torch.tensor([[0.0] + [500.0] * 48] * 2),
            torch.randn(2, 1, 8, 3),
            torch.randn(2, 1, 3),
            torch.randn(2, 1, 3),
            utonia_features=features,
        )
    finally:
        handle.remove()

    assert output.shape == (2, 48, 1, 8, 3)
    assert captured["tokens"].shape == (2, 49, 8, 69)


def test_history_model_predicts_45_future_frames_from_four_clean_history_frames():
    model = make_tiny_model(
        objective_type="ddpm", conditioning="history", history_frames=4
    )

    output = model(
        torch.randn(2, 45, 1, 8, 3),
        torch.tensor([[0.0] * 4 + [500.0] * 45] * 2),
        torch.randn(2, 4, 1, 8, 3),
    )

    assert output.shape == (2, 45, 1, 8, 3)


def test_history_model_broadcasts_utonia_features_to_all_49_temporal_tokens():
    model = make_tiny_model(
        objective_type="ddpm",
        utonia_feature_dim=5,
        conditioning="history",
        history_frames=4,
    )
    captured = {}
    handle = model.utonia_feature_projection.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("tokens", inputs[0].detach().clone())
    )
    features = torch.randn(2, 8, 5)
    try:
        output = model(
            torch.randn(2, 45, 1, 8, 3),
            torch.tensor([[0.0] * 4 + [500.0] * 45] * 2),
            torch.randn(2, 4, 1, 8, 3),
            utonia_features=features,
        )
    finally:
        handle.remove()

    assert output.shape == (2, 45, 1, 8, 3)
    assert captured["tokens"].shape == (2, 49, 8, 69)
    assert torch.equal(
        captured["tokens"][:, :1, :, 64:].expand_as(captured["tokens"][:, :, :, 64:]),
        captured["tokens"][:, :, :, 64:],
    )


@pytest.mark.parametrize(
    "features, message",
    [
        (None, "utonia_features must be provided"),
        (torch.zeros(1, 8, 5), "batch size"),
        (torch.zeros(2, 7, 5), "point count"),
        (torch.zeros(2, 8, 4), "feature width"),
    ],
)
def test_model_rejects_invalid_utonia_features(features, message):
    model = make_tiny_model(utonia_feature_dim=5)

    with pytest.raises(ValueError, match=message):
        model(
            torch.zeros(2, 48, 1, 8, 3),
            torch.tensor([[0.0] + [500.0] * 48] * 2),
            torch.zeros(2, 1, 8, 3),
            torch.zeros(2, 1, 3),
            torch.zeros(2, 1, 3),
            utonia_features=features,
        )
