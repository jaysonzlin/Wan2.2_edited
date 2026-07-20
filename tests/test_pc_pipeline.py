from types import SimpleNamespace

import torch

from wan.pc_pipeline import PCDDIMPipeline, PCFlowPipeline


class ZeroFlowModel(torch.nn.Module):
    n_future_frames = 48

    def forward(self, noisy, frame_times, init_pc, linear, angular):
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


class RecordingDDPMModel(torch.nn.Module):
    n_future_frames = 48

    def __init__(self):
        super().__init__()
        self.frame_times = []

    def forward(self, noisy, frame_times, init_pc, linear, angular):
        self.frame_times.append(frame_times.detach().clone())
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
