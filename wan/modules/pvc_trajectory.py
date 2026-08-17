"""History DDPM trajectory model with persistent frame-local point-view tokens."""

import torch
import torch.nn as nn

from .pc_trajectory import PCTrajectoryModel


class PVCTrajectoryModel(PCTrajectoryModel):
    """Specialized history model whose view stream conditions spatial attention."""

    def __init__(self, *args, utonia_feature_dim: int | None = None, **kwargs):
        super().__init__(
            *args,
            objective_type="ddpm",
            conditioning="history",
            history_frames=4,
            utonia_feature_dim=utonia_feature_dim,
            **kwargs,
        )
        if utonia_feature_dim is None:
            raise ValueError("PVC requires utonia_feature_dim")
        self.point_view_type_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, self.latent_dim)
        )

    def forward(
        self,
        noisy_future_state: torch.Tensor,
        frame_times: torch.Tensor,
        points_history: torch.Tensor,
        point_views: torch.Tensor,
        point_view_mask: torch.Tensor,
        utonia_features: torch.Tensor,
        point_view_utonia_features: torch.Tensor,
    ) -> torch.Tensor:
        points, _, temb = super().encode_states(
            noisy_future_state,
            frame_times,
            points_history,
            None,
            None,
            utonia_features,
        )
        batch = points.shape[0]
        expected_views = (batch, 49, self.n_points, 3)
        if point_views.shape != expected_views:
            raise ValueError(f"point_views must have shape {expected_views}")
        expected_mask = (batch, 49, self.n_points)
        if point_view_mask.dtype != torch.bool or point_view_mask.shape != expected_mask:
            raise ValueError(f"point_view_mask must have shape {expected_mask} and dtype bool")
        expected_features = (batch, 49, self.n_points, self.utonia_feature_dim)
        if point_view_utonia_features.shape != expected_features:
            raise ValueError(f"point_view_utonia_features must have shape {expected_features}")
        view_states = self.input_encoder(point_views.reshape(-1, self.n_points, 3))
        view_states = view_states.reshape(batch, 49, self.n_points, self.latent_dim)
        view_features = self.utonia_feature_norm(
            point_view_utonia_features.to(device=points.device, dtype=points.dtype)
        )
        view_states = self.utonia_feature_projection(
            torch.cat((view_states, view_features), dim=-1)
        )
        positions = self.position_embedding[:, 2:].to(device=points.device, dtype=points.dtype)
        positions = positions.reshape(1, 49, self.n_points, self.latent_dim)
        temporal_width = self.latent_dim // 4
        view_positions = torch.cat(
            (positions[..., :temporal_width], torch.zeros_like(positions[..., temporal_width:])),
            dim=-1,
        )
        view_states = view_states + view_positions + self.point_view_type_embedding
        for block in self.blocks:
            points, view_states = block.forward_with_point_views(
                points, view_states, point_view_mask, temb
            )
        return self.decode_states(points, temb, points_history)
