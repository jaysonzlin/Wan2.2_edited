from types import SimpleNamespace

import torch

from wan.pc_pipeline import (
    PCDDIMPipeline,
    PCFlowPipeline,
    PCHistoryDDIMPipeline,
    PCHistoryFlowPipeline,
)


class ZeroFlowModel(torch.nn.Module):
    n_future_frames = 48

    def __init__(self):
        super().__init__()
        self.utonia_conditions = []

    def forward(self, noisy, frame_times, init_pc, linear, angular, utonia_features=None):
        self.utonia_conditions.append(utonia_features)
        return torch.zeros_like(noisy)


class FakeFlowScheduler:
    def set_timesteps(self, num_inference_steps, device, shift):
        self.timesteps = torch.arange(num_inference_steps, device=device)
        self.shift = shift

    def step(self, flow, timestep, sample, return_dict=True, generator=None):
        return SimpleNamespace(prev_sample=torch.zeros_like(sample))


def test_pipeline_adds_source_only_after_integration():
    scheduler = FakeFlowScheduler()
    pipeline = PCFlowPipeline(ZeroFlowModel(), scheduler, time_shift=5.0)

    output = pipeline(
        torch.full((1, 1, 2, 3), 7.0),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        "cpu",
        2,
        torch.Generator().manual_seed(0),
    )

    assert scheduler.shift == 5.0
    assert output.shape == (1, 48, 1, 2, 3)
    assert torch.allclose(output, torch.full_like(output, 7.0))


def test_flow_pipeline_forwards_utonia_features_to_every_model_step():
    model = ZeroFlowModel()
    pipeline = PCFlowPipeline(model, FakeFlowScheduler(), time_shift=5.0)
    features = torch.randn(1, 2, 5)

    pipeline(
        torch.zeros(1, 1, 2, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        "cpu",
        2,
        utonia_features=features,
    )

    assert len(model.utonia_conditions) == 2
    assert all(torch.equal(condition, features) for condition in model.utonia_conditions)


class RecordingDDPMModel(torch.nn.Module):
    n_future_frames = 48

    def __init__(self):
        super().__init__()
        self.frame_times = []

    def forward(self, noisy, frame_times, init_pc, linear, angular, utonia_features=None):
        self.frame_times.append(frame_times.detach().clone())
        self.utonia_conditions = getattr(self, "utonia_conditions", []) + [utonia_features]
        return torch.zeros_like(noisy)


class FakeDDIMScheduler:
    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = torch.tensor([9, 3], device=device)

    def step(self, prediction, timestep, sample, generator=None):
        return SimpleNamespace(prev_sample=torch.zeros_like(sample))


def test_ddim_pipeline_uses_one_timestep_for_all_49_tokens_and_returns_absolute_state():
    model = RecordingDDPMModel()
    pipeline = PCDDIMPipeline(model, FakeDDIMScheduler())

    output = pipeline(
        torch.full((1, 1, 2, 3), 7.0),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        "cpu",
        2,
        torch.Generator().manual_seed(0),
    )

    assert len(model.frame_times) == 2
    assert all(torch.equal(times, torch.full_like(times, value)) for times, value in zip(model.frame_times, (9, 3)))
    assert torch.equal(output, torch.zeros_like(output))


def test_ddim_pipeline_forwards_utonia_features_to_every_model_step():
    model = RecordingDDPMModel()
    pipeline = PCDDIMPipeline(model, FakeDDIMScheduler())
    features = torch.randn(1, 2, 5)

    pipeline(
        torch.zeros(1, 1, 2, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        "cpu",
        2,
        utonia_features=features,
    )

    assert len(model.utonia_conditions) == 2
    assert all(torch.equal(condition, features) for condition in model.utonia_conditions)


class RecordingHistoryModel(torch.nn.Module):
    n_future_frames = 45

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy, frame_times, points_history, utonia_features=None):
        self.calls.append(
            (
                frame_times.detach().clone(),
                points_history.detach().clone(),
                utonia_features,
            )
        )
        return torch.zeros_like(noisy)


def test_history_ddim_pipeline_forwards_four_known_frames_and_utonia():
    model = RecordingHistoryModel()
    pipeline = PCHistoryDDIMPipeline(model, FakeDDIMScheduler())
    history = torch.stack(
        [torch.full((1, 1, 2, 3), value) for value in (1.0, 2.0, 3.0, 4.0)],
        dim=1,
    )
    features = torch.randn(1, 2, 5)

    output = pipeline(
        history,
        "cpu",
        2,
        torch.Generator().manual_seed(0),
        utonia_features=features,
    )

    assert output.shape == (1, 45, 1, 2, 3)
    assert len(model.calls) == 2
    for (frame_times, known_frames, condition), timestep in zip(
        model.calls, (9.0, 3.0)
    ):
        assert torch.equal(known_frames, history)
        assert condition is features
        assert torch.equal(frame_times[:, :4], torch.zeros(1, 4))
        assert torch.equal(frame_times[:, 4:], torch.full((1, 45), timestep))


def test_history_flow_pipeline_anchors_generated_displacements_to_frame_zero():
    model = RecordingHistoryModel()
    scheduler = FakeFlowScheduler()
    pipeline = PCHistoryFlowPipeline(model, scheduler, time_shift=5.0)
    history = torch.stack(
        [torch.full((1, 1, 2, 3), value) for value in (7.0, 8.0, 9.0, 10.0)],
        dim=1,
    )

    output = pipeline(history, "cpu", 2, torch.Generator().manual_seed(0))

    assert scheduler.shift == 5.0
    assert output.shape == (1, 45, 1, 2, 3)
    assert torch.equal(output, torch.full_like(output, 7.0))
    assert all(torch.equal(call[1], history) for call in model.calls)
    assert torch.equal(model.calls[1][0][:, :4], torch.zeros(1, 4))
    assert torch.equal(model.calls[1][0][:, 4:], torch.ones(1, 45))
