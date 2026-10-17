"""Native Wan-only flow sampling for the vanilla-video ablation."""

from dataclasses import dataclass

import torch

from training.wan_i2v_training import expand_latent_timesteps, pin_history_latents


@dataclass(frozen=True)
class WanVideoSample:
    """The final latent produced by a native Wan sampling trajectory."""

    video_latent: torch.Tensor


class WanVideoPipeline:
    """Advance only Wan's video state with a Flow-UniPC scheduler."""

    def __init__(self, model, scheduler, time_shift: float) -> None:
        if time_shift <= 0:
            raise ValueError("time_shift must be positive")
        self.model = model
        self.scheduler = scheduler
        self.time_shift = time_shift

    @staticmethod
    def _prev_sample(step_output):
        return step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]

    @torch.no_grad()
    def __call__(
        self,
        *,
        condition_latent: torch.Tensor,
        video_shape: tuple[int, int, int, int],
        context: list[torch.Tensor],
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> WanVideoSample:
        """Sample a video while retaining the clean latent history frames."""
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if condition_latent.ndim != 4:
            raise ValueError("condition_latent must have shape [C, H, Ht, Wt]")
        history_frames = condition_latent.shape[1]
        if condition_latent.shape != (
            video_shape[0], history_frames, video_shape[2], video_shape[3]
        ):
            raise ValueError("condition_latent must match the video shape and history prefix")
        if not 0 < history_frames < video_shape[1]:
            raise ValueError("history frames must leave at least one sampled frame")

        device = condition_latent.device
        video_state = torch.randn(
            video_shape, device=device, dtype=condition_latent.dtype, generator=generator
        )
        video_state = pin_history_latents(video_state, torch.cat((condition_latent, video_state[:, history_frames:]), dim=1), history_frames)
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=self.time_shift)
        for timestep in self.scheduler.timesteps:
            frame_times = torch.full(
                (1, video_shape[1]), timestep.item(), device=device, dtype=video_state.dtype
            )
            frame_times[:, :history_frames] = 0
            prediction = self.model(
                [video_state],
                expand_latent_timesteps(frame_times, video_shape[2], video_shape[3]),
                context,
                video_shape[1] * (video_shape[2] // 2) * (video_shape[3] // 2),
            )[0]
            video_state = self._prev_sample(
                self.scheduler.step(
                    prediction.unsqueeze(0),
                    timestep,
                    video_state.unsqueeze(0),
                    return_dict=True,
                    generator=generator,
                )
            ).squeeze(0)
            video_state = pin_history_latents(
                video_state,
                torch.cat((condition_latent, video_state[:, history_frames:]), dim=1),
                history_frames,
            )
        return WanVideoSample(video_latent=video_state)
