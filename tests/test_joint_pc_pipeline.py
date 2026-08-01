import torch

from wan.joint_pc_pipeline import JointWanPhysCtrlPipeline


class _Scheduler:
    def __init__(self, timesteps):
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


class _JointModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        video = kwargs["video_x"][0]
        points = kwargs["noisy_future_state"]
        return [torch.zeros_like(video)], torch.zeros_like(points)


def test_joint_pipeline_evaluates_both_branches_once_per_synchronized_outer_step():
    model = _JointModel()
    video_scheduler = _Scheduler([9, 5, 1])
    pc_scheduler = _Scheduler([999, 500, 0])
    pipeline = JointWanPhysCtrlPipeline(model, video_scheduler, pc_scheduler, time_shift=5.0)
    condition = torch.full((2, 1, 2, 2), 7.0)
    init_pc = torch.full((2, 1, 3, 3), 4.0)
    velocities = torch.zeros((2, 1, 3))

    result = pipeline(
        condition_latent=condition,
        video_shape=(2, 3, 2, 2),
        context=[torch.zeros(1, 4)],
        initial_point_clouds=init_pc,
        initial_linear_velocities=velocities,
        initial_angular_velocities=velocities,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(model.calls) == 3
    assert len(video_scheduler.step_calls) == len(pc_scheduler.step_calls) == 3
    assert video_scheduler.set_calls == [(3, {"shift": 5.0})]
    assert pc_scheduler.set_calls == [(3, {})]
    assert torch.equal(result.video_latent[:, :1], condition)
    assert result.future_point_clouds.shape == (1, 2, 48, 1, 3, 3)
    assert model.calls[0]["frame_times"].shape == (1, 2, 49)
    assert model.calls[0]["init_pc"].eq(init_pc.unsqueeze(0)).all()
    assert model.calls[0]["video_t"].shape == (1, 3)
