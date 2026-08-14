import pytest
import torch

from training.pc_objectives import make_pc_flow_batch


def test_flow_batch_uses_displacements_and_source_time_zero():
    source = torch.full((1, 1, 2, 3), 10.0)
    future = torch.full((1, 48, 1, 2, 3), 11.0)

    batch = make_pc_flow_batch(
        future, source, torch.Generator().manual_seed(0), 5.0, 1000
    )

    assert batch.model_input.shape == future.shape
    assert batch.velocity_target.shape == future.shape
    assert torch.equal(batch.frame_times[:, :1], torch.zeros(1, 1))
    assert torch.all(batch.frame_times[:, 1:] > 0)


def test_flow_target_is_noise_minus_displacement(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, **kwargs: torch.full(shape, 3.0, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    batch = make_pc_flow_batch(
        torch.ones(1, 48, 1, 1, 3),
        torch.zeros(1, 1, 1, 3),
        torch.Generator().manual_seed(0),
        1.0,
        1000,
    )

    assert torch.equal(batch.velocity_target, torch.full((1, 48, 1, 1, 3), 2.0))


def test_flow_batch_uses_zero_times_for_four_known_history_frames():
    history = torch.full((1, 4, 1, 2, 3), 10.0)
    future = torch.full((1, 45, 1, 2, 3), 11.0)

    batch = make_pc_flow_batch(
        future,
        history,
        torch.Generator().manual_seed(0),
        5.0,
        1000,
        known_frames=4,
    )

    assert batch.model_input.shape == future.shape
    assert batch.velocity_target.shape == future.shape
    assert torch.equal(batch.frame_times[:, :4], torch.zeros(1, 4))
    assert torch.all(batch.frame_times[:, 4:] > 0)


@pytest.mark.parametrize(
    "future_frames, known_frames",
    [(48, 4), (45, 1), (45, 2), (45, 5), (48, True)],
)
def test_flow_batch_rejects_unsupported_temporal_layouts(
    future_frames, known_frames
):
    future = torch.zeros(1, future_frames, 1, 2, 3)
    known = 4 if known_frames == 4 else 1
    source = torch.zeros(1, known, 1, 2, 3)

    with pytest.raises(ValueError):
        make_pc_flow_batch(
            future,
            source,
            torch.Generator().manual_seed(0),
            5.0,
            1000,
            known_frames=known_frames,
        )
