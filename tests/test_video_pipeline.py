import torch

from wan.video_pipeline import WanVideoPipeline


class _Scheduler:
    def __init__(self, timesteps=(9, 5)):
        self._timesteps = torch.tensor(timesteps)
        self.timesteps = None
        self.set_calls = []
        self.step_calls = []

    def set_timesteps(self, num_steps, device, **kwargs):
        self.set_calls.append((num_steps, kwargs))
        self.timesteps = self._timesteps.to(device)

    def step(self, prediction, timestep, sample, **kwargs):
        self.step_calls.append((prediction.clone(), timestep, sample.clone()))
        return type("Step", (), {"prev_sample": sample - 1})()


class _RecordingWan:
    def __init__(self):
        self.calls = []

    def __call__(self, x, t, context, seq_len, y=None):
        self.calls.append({"x": x, "t": t, "context": context, "seq_len": seq_len, "y": y})
        return [torch.zeros_like(x[0])]


def test_video_pipeline_calls_native_wan_only_and_repins_history():
    model = _RecordingWan()
    scheduler = _Scheduler()
    pipeline = WanVideoPipeline(model, scheduler, time_shift=5.0)
    history = torch.full((2, 1, 2, 2), 7.0)

    result = pipeline(
        condition_latent=history,
        video_shape=(2, 3, 2, 2),
        context=[torch.zeros(1, 4)],
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert scheduler.set_calls == [(2, {"shift": 5.0})]
    assert len(scheduler.step_calls) == len(model.calls) == 2
    assert all(set(call) == {"x", "t", "context", "seq_len", "y"} for call in model.calls)
    assert all(call["y"] is None for call in model.calls)
    assert model.calls[0]["x"][0].shape == (2, 3, 2, 2)
    assert model.calls[0]["t"].shape == (1, 3)
    assert torch.equal(result.video_latent[:, :1], history)
    assert torch.equal(scheduler.step_calls[1][2][0, :, :1], history)


def test_video_pipeline_is_repeatable_for_a_fixed_generator_seed():
    history = torch.zeros((1, 1, 2, 2))

    first = WanVideoPipeline(_RecordingWan(), _Scheduler(), time_shift=5.0)(
        condition_latent=history,
        video_shape=(1, 3, 2, 2),
        context=[torch.zeros(1, 4)],
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(8),
    )
    second = WanVideoPipeline(_RecordingWan(), _Scheduler(), time_shift=5.0)(
        condition_latent=history,
        video_shape=(1, 3, 2, 2),
        context=[torch.zeros(1, 4)],
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(8),
    )

    torch.testing.assert_close(first.video_latent, second.video_latent)
