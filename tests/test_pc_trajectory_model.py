import pytest
import torch

from wan.modules.pc_trajectory import PCTrajectoryModel


def make_tiny_model(objective_type="flow"):
    return PCTrajectoryModel(
        n_points=8,
        n_future_frames=48,
        latent_dim=64,
        n_layers=1,
        num_heads=1,
        point_embed=False,
        objective_type=objective_type,
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
