from types import SimpleNamespace

import torch

from wan.pc_pipeline import PCFlowPipeline


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
