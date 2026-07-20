"""Sampling pipeline for point-cloud flow trajectories."""

import torch


class PCFlowPipeline:
    """Integrate PC flow predictions and convert displacements to positions."""

    def __init__(self, model, scheduler, time_shift: float):
        if time_shift <= 0:
            raise ValueError("time_shift must be positive")
        self.model = model
        self.scheduler = scheduler
        self.time_shift = time_shift

    @torch.no_grad()
    def __call__(
        self,
        init_pc: torch.Tensor,
        initial_linear_velocity: torch.Tensor,
        initial_angular_velocity: torch.Tensor,
        device: str | torch.device,
        num_inference_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = torch.device(device)
        init_pc = init_pc.to(device)
        initial_linear_velocity = initial_linear_velocity.to(device)
        initial_angular_velocity = initial_angular_velocity.to(device)
        batch_size, _, n_points, _ = init_pc.shape
        sample = torch.randn(
            (batch_size, self.model.n_future_frames, 1, n_points, 3),
            device=device,
            dtype=init_pc.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(
            num_inference_steps, device=device, shift=self.time_shift
        )
        for timestep in self.scheduler.timesteps:
            frame_times = torch.full(
                (batch_size, self.model.n_future_frames + 1),
                timestep.item(),
                device=device,
                dtype=sample.dtype,
            )
            frame_times[:, 0] = 0
            flow = self.model(
                sample,
                frame_times,
                init_pc,
                initial_linear_velocity,
                initial_angular_velocity,
            )
            sample = self.scheduler.step(
                flow, timestep, sample, return_dict=True, generator=generator
            ).prev_sample
        return sample + init_pc.unsqueeze(1)


class PCDDIMPipeline:
    """Sample absolute future positions from a DDPM x0 PC model."""

    def __init__(self, model, scheduler):
        self.model, self.scheduler = model, scheduler

    @torch.no_grad()
    def __call__(self, init_pc, initial_linear_velocity, initial_angular_velocity, device, num_inference_steps, generator=None):
        device = torch.device(device)
        init_pc = init_pc.to(device)
        linear = initial_linear_velocity.to(device)
        angular = initial_angular_velocity.to(device)
        batch_size, _, n_points, _ = init_pc.shape
        sample = torch.randn((batch_size, self.model.n_future_frames, 1, n_points, 3), device=device, dtype=init_pc.dtype, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            frame_times = torch.full((batch_size, self.model.n_future_frames + 1), timestep.item(), device=device, dtype=sample.dtype)
            prediction = self.model(sample, frame_times, init_pc, linear, angular)
            sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
        return sample
