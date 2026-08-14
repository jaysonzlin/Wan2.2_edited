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
