"""DDIM sampling for history-conditioned PVC trajectory models."""

import torch


class PVCHistoryDDIMPipeline:
    def __init__(self, model, scheduler):
        self.model, self.scheduler = model, scheduler

    @torch.no_grad()
    def __call__(
        self, points_history, point_views, point_view_mask, utonia_features,
        point_view_utonia_features, device, num_inference_steps, generator=None,
    ):
        device = torch.device(device)
        points_history = points_history.to(device)
        point_views = point_views.to(device)
        point_view_mask = point_view_mask.to(device)
        utonia_features = utonia_features.to(device)
        point_view_utonia_features = point_view_utonia_features.to(device)
        batch, history_frames, _, point_count, _ = points_history.shape
        sample = torch.randn(
            (batch, self.model.n_future_frames, 1, point_count, 3),
            device=device, dtype=points_history.dtype, generator=generator,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            times = torch.full(
                (batch, history_frames + self.model.n_future_frames), timestep.item(),
                device=device, dtype=sample.dtype,
            )
            times[:, :history_frames] = 0
            prediction = self.model(
                sample, times, points_history, point_views, point_view_mask,
                utonia_features, point_view_utonia_features,
            )
            sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
        return sample
