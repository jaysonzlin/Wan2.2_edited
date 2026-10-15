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
        initial_linear_velocities: torch.Tensor | None,
        initial_angular_velocities: torch.Tensor | None,
        utonia_features: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
        pc_generator: torch.Generator | None = None,
        advance_pc_state: bool = True,
    ) -> JointSample:
        """Sample I2V and, optionally, advance the auxiliary PC DDIM trajectory.

        ``advance_pc_state=False`` preserves one initial PC-noise state throughout
        video sampling.  The retained PC encoder and bridge path receives the
        current video timestep as its conditioning time, but PC scheduler setup,
        decoding, and DDIM feedback are all disabled.
        """
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        history_frames = condition_latent.shape[1]
        if condition_latent.shape != (video_shape[0], history_frames, video_shape[2], video_shape[3]):
            raise ValueError("condition_latent must have shape [C, H, Ht, Wt]")
        if initial_point_clouds.ndim != 4 or initial_point_clouds.shape[1] != history_frames:
            raise ValueError("initial_point_clouds must have matching [K, H, N, 3] history")
        device = condition_latent.device
        object_count, _, point_count, _ = initial_point_clouds.shape
        future_frame_count = self.model.pc_model.n_future_frames
        pc_generator = generator if pc_generator is None else pc_generator
        video_state = torch.randn(video_shape, device=device, dtype=condition_latent.dtype, generator=generator)
        video_state[:, :history_frames] = condition_latent
        point_state = torch.randn(
            (1, object_count, future_frame_count, 1, point_count, 3),
            device=device,
            dtype=initial_point_clouds.dtype,
            generator=pc_generator,
        )
        init_pc = initial_point_clouds.unsqueeze(0).unsqueeze(3).to(device)
        linear = None if initial_linear_velocities is None else initial_linear_velocities.unsqueeze(0).to(device)
        angular = None if initial_angular_velocities is None else initial_angular_velocities.unsqueeze(0).to(device)
        features = None if utonia_features is None else utonia_features.unsqueeze(0).to(device)
        self.video_scheduler.set_timesteps(num_inference_steps, device=device, shift=self.time_shift)
        if advance_pc_state:
            self.pc_scheduler.set_timesteps(num_inference_steps, device=device)
            if len(self.video_scheduler.timesteps) != len(self.pc_scheduler.timesteps):
                raise ValueError("video and PC schedulers must expose the same number of timesteps")
            timestep_pairs = zip(self.video_scheduler.timesteps, self.pc_scheduler.timesteps)
        else:
            timestep_pairs = (
                (video_timestep, video_timestep)
                for video_timestep in self.video_scheduler.timesteps
            )
        for video_timestep, pc_timestep in timestep_pairs:
            video_frame_times = torch.full(
                (1, video_shape[1]), video_timestep.item(), device=device, dtype=video_state.dtype
            )
            video_frame_times[:, :history_frames] = 0
            frame_times = torch.full(
                (1, object_count, future_frame_count + history_frames),
                pc_timestep.item(),
                device=device,
                dtype=point_state.dtype,
            )
            frame_times[:, :, :history_frames] = 0
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
                utonia_features=features,
                return_pc_output=advance_pc_state,
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
            video_state[:, :history_frames] = condition_latent
            if advance_pc_state:
                point_state = self._prev_sample(
                    self.pc_scheduler.step(
                        pc_prediction, pc_timestep, point_state, return_dict=True, generator=pc_generator
                    )
                )
        return JointSample(video_latent=video_state, future_point_clouds=point_state)
