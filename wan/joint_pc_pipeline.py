"""Synchronized Wan flow and per-object PC DDIM sampling."""

from dataclasses import dataclass

import torch

from training.wan_i2v_training import expand_latent_timesteps


@dataclass(frozen=True)
class JointSample:
    """Final video latent and absolute future point-cloud trajectories."""

    video_latent: torch.Tensor
    future_point_clouds: torch.Tensor


class JointWanPhysCtrlPipeline:
    """Couple one Flow-UniPC update and one DDIM update at every outer step."""

    def __init__(self, model, video_scheduler, pc_scheduler, time_shift: float) -> None:
        if time_shift <= 0:
            raise ValueError("time_shift must be positive")
        self.model = model
        self.video_scheduler = video_scheduler
        self.pc_scheduler = pc_scheduler
        self.time_shift = time_shift

    @staticmethod
    def _prev_sample(step_output):
        return step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]

    @torch.no_grad()
    def __call__(
        self,
        condition_latent: torch.Tensor,
        video_shape: tuple[int, int, int, int],
        context: list[torch.Tensor],
        initial_point_clouds: torch.Tensor,
        initial_linear_velocities: torch.Tensor,
        initial_angular_velocities: torch.Tensor,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> JointSample:
        """Jointly sample one I2V latent and K PC trajectories with equal outer-step counts."""
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if condition_latent.shape != (video_shape[0], 1, video_shape[2], video_shape[3]):
            raise ValueError("condition_latent must be the clean first video latent slot")
        if initial_point_clouds.ndim != 4 or initial_point_clouds.shape[1] != 1:
            raise ValueError("initial_point_clouds must have shape [K, 1, N, 3]")
        device = condition_latent.device
        object_count, _, point_count, _ = initial_point_clouds.shape
        future_frame_count = self.model.pc_model.n_future_frames
        video_state = torch.randn(video_shape, device=device, dtype=condition_latent.dtype, generator=generator)
        video_state[:, :1] = condition_latent
        point_state = torch.randn(
            (1, object_count, future_frame_count, 1, point_count, 3),
            device=device,
            dtype=initial_point_clouds.dtype,
            generator=generator,
        )
        init_pc = initial_point_clouds.unsqueeze(0).to(device)
        linear = initial_linear_velocities.unsqueeze(0).to(device)
        angular = initial_angular_velocities.unsqueeze(0).to(device)
        self.video_scheduler.set_timesteps(num_inference_steps, device=device, shift=self.time_shift)
        self.pc_scheduler.set_timesteps(num_inference_steps, device=device)
        if len(self.video_scheduler.timesteps) != len(self.pc_scheduler.timesteps):
            raise ValueError("video and PC schedulers must expose the same number of timesteps")
        for video_timestep, pc_timestep in zip(self.video_scheduler.timesteps, self.pc_scheduler.timesteps):
            video_frame_times = torch.full(
                (1, video_shape[1]), video_timestep.item(), device=device, dtype=video_state.dtype
            )
            video_frame_times[:, 0] = 0
            frame_times = torch.full(
                (1, object_count, future_frame_count + 1),
                pc_timestep.item(),
                device=device,
                dtype=point_state.dtype,
            )
            video_prediction, pc_prediction = self.model(
                video_x=[video_state],
                video_t=expand_latent_timesteps(
                    video_frame_times, video_shape[2], video_shape[3]
                ),
                context=context,
                seq_len=video_shape[1] * (video_shape[2] // 2) * (video_shape[3] // 2),
                noisy_future_state=point_state,
                frame_times=frame_times,
                init_pc=init_pc,
                initial_linear_velocity=linear,
                initial_angular_velocity=angular,
            )
            video_state = self._prev_sample(
                self.video_scheduler.step(
                    video_prediction[0].unsqueeze(0),
                    video_timestep,
                    video_state.unsqueeze(0),
                    return_dict=True,
                    generator=generator,
                )
            ).squeeze(0)
            video_state[:, :1] = condition_latent
            point_state = self._prev_sample(
                self.pc_scheduler.step(
                    pc_prediction, pc_timestep, point_state, return_dict=True, generator=generator
                )
            )
        return JointSample(video_latent=video_state, future_point_clouds=point_state)
