from types import SimpleNamespace

import torch

from wan.pvc_pipeline import PVCHistoryDDIMPipeline


class Model(torch.nn.Module):
    n_future_frames = 45

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy, times, history, views, mask, features, view_features):
        self.calls.append((times, history, views, mask, features, view_features))
        return torch.zeros_like(noisy)


class Scheduler:
    def set_timesteps(self, steps, device): self.timesteps = torch.tensor([9, 3], device=device)
    def step(self, prediction, timestep, sample, generator=None): return SimpleNamespace(prev_sample=torch.zeros_like(sample))


def test_pvc_pipeline_forwards_full_view_condition_every_denoising_step():
    model = Model()
    pipeline = PVCHistoryDDIMPipeline(model, Scheduler())
    history = torch.randn(1, 4, 1, 2, 3)
    views = torch.randn(1, 49, 2, 3)
    mask = torch.ones(1, 49, 2, dtype=torch.bool)
    features = torch.randn(1, 2, 5)
    view_features = torch.randn(1, 49, 2, 5)

    output = pipeline(history, views, mask, features, view_features, "cpu", 2)

    assert output.shape == (1, 45, 1, 2, 3)
    assert len(model.calls) == 2
    assert all(call[2] is views and call[3] is mask for call in model.calls)
    assert torch.equal(model.calls[0][0][:, :4], torch.zeros(1, 4))
