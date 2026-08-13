"""PhysCtrl-equivalent point-cloud trajectory diffusion model."""

import math

import torch
import torch.nn as nn

from .pc_physctrl import (
    PhysCtrlOutputHead,
    PhysCtrlSpatialTemporalBlock,
    PhysCtrlTimestepEmbedding,
    physctrl_position_embedding,
)


class PointEmbed(nn.Module):
    """The 96-feature Fourier XYZ point encoder used by PhysCtrl PC-DiT."""

    def __init__(self, dim: int, hidden_dim: int = 96):
        super().__init__()
        basis = torch.pow(2, torch.arange(hidden_dim // 6)).float() * math.pi
        basis = torch.stack(
            (
                torch.cat((basis, torch.zeros_like(basis), torch.zeros_like(basis))),
                torch.cat((torch.zeros_like(basis), basis, torch.zeros_like(basis))),
                torch.cat((torch.zeros_like(basis), torch.zeros_like(basis), basis)),
            )
        )
        self.register_buffer("basis", basis)
        self.projection = nn.Linear(hidden_dim + 3, dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bnd,de->bne", points, self.basis)
        embedding = torch.cat((projections.sin(), projections.cos(), points), dim=-1)
        return self.projection(embedding)


class PCTrajectoryModel(nn.Module):
    """Predict 48 future point-cloud frames from initial-state conditions."""

    def __init__(
        self,
        n_points: int = 2048,
        n_future_frames: int = 48,
        latent_dim: int = 256,
        n_layers: int = 8,
        num_heads: int = 4,
        point_embed: bool = True,
        objective_type: str = "flow",
        utonia_feature_dim: int | None = None,
    ):
        super().__init__()
        if objective_type not in {"flow", "ddpm"}:
            raise ValueError("objective_type must be 'flow' or 'ddpm'")
        if latent_dim % 64:
            raise ValueError("latent_dim must be divisible by 64")
        if num_heads != latent_dim // 64:
            raise ValueError("num_heads must equal latent_dim // 64")
        if not point_embed:
            raise ValueError("point_embed must be true")
        if utonia_feature_dim is not None and utonia_feature_dim <= 0:
            raise ValueError("utonia_feature_dim must be positive")

        self.objective_type = objective_type
        self.n_points = n_points
        self.n_future_frames = n_future_frames
        self.latent_dim = latent_dim
        self.utonia_feature_dim = utonia_feature_dim
        self.input_encoder = PointEmbed(latent_dim)
        if utonia_feature_dim is not None:
            self.utonia_feature_norm = nn.LayerNorm(utonia_feature_dim)
            self.utonia_feature_projection = nn.Linear(
                latent_dim + utonia_feature_dim, latent_dim
            )
        self.linear_velocity_encoder = nn.Linear(3, latent_dim)
        self.angular_velocity_encoder = nn.Linear(3, latent_dim)
        self.time_embedding = PhysCtrlTimestepEmbedding(latent_dim)
        self.blocks = nn.ModuleList(
            PhysCtrlSpatialTemporalBlock(latent_dim, num_heads)
            for _ in range(n_layers)
        )
        self.output_head = PhysCtrlOutputHead(latent_dim)
        self.register_buffer(
            "position_embedding",
            physctrl_position_embedding(n_points, n_future_frames + 1, latent_dim),
            persistent=False,
        )

    def forward(
        self,
        noisy_future_state: torch.Tensor,
        frame_times: torch.Tensor,
        init_pc: torch.Tensor,
        initial_linear_velocity: torch.Tensor,
        initial_angular_velocity: torch.Tensor,
        utonia_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points, controls, temb = self.encode_states(
            noisy_future_state,
            frame_times,
            init_pc,
            initial_linear_velocity,
            initial_angular_velocity,
            utonia_features,
        )
        for block in self.blocks:
            points, controls = block(points, controls, temb)
        return self.decode_states(points, temb, init_pc)

    def encode_states(
        self,
        noisy_future_state: torch.Tensor,
        frame_times: torch.Tensor,
        init_pc: torch.Tensor,
        initial_linear_velocity: torch.Tensor,
        initial_angular_velocity: torch.Tensor,
        utonia_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a trajectory into point/control states for externally interleaved blocks."""
        batch = noisy_future_state.shape[0]
        expected = (batch, self.n_future_frames, 1, self.n_points, 3)
        if noisy_future_state.shape != expected:
            raise ValueError(f"noisy_future_state must have shape {expected}")
        if init_pc.shape != (batch, 1, self.n_points, 3):
            raise ValueError("init_pc must have shape (B, 1, N, 3)")
        if frame_times.shape != (batch, self.n_future_frames + 1):
            raise ValueError("frame_times must have shape (B, 49)")
        if self.objective_type == "flow" and not torch.equal(
            frame_times[:, 0], torch.zeros_like(frame_times[:, 0])
        ):
            raise ValueError("frame_times[:, 0] must be zero")
        if (
            initial_linear_velocity.shape != (batch, 1, 3)
            or initial_angular_velocity.shape != (batch, 1, 3)
        ):
            raise ValueError("initial velocities must have shape (B, 1, 3)")
        if self.utonia_feature_dim is not None:
            if utonia_features is None:
                raise ValueError("utonia_features must be provided when fusion is configured")
            if utonia_features.ndim != 3:
                raise ValueError("utonia_features must have shape (B, N, D)")
            if utonia_features.shape[0] != batch:
                raise ValueError("utonia_features batch size must match trajectory input")
            if utonia_features.shape[1] != self.n_points:
                raise ValueError("utonia_features point count must match trajectory input")
            if utonia_features.shape[2] != self.utonia_feature_dim:
                raise ValueError("utonia_features feature width must match configured fusion")

        future_positions = (
            init_pc.unsqueeze(1) + noisy_future_state
            if self.objective_type == "flow"
            else noisy_future_state
        )
        coordinates = torch.cat((init_pc.unsqueeze(1), future_positions), dim=1).squeeze(2)
        points = self.input_encoder(coordinates.reshape(-1, self.n_points, 3))
        points = points.reshape(
            batch,
            self.n_future_frames + 1,
            self.n_points,
            self.latent_dim,
        )
        if self.utonia_feature_dim is not None:
            assert utonia_features is not None
            feature_tokens = self.utonia_feature_norm(
                utonia_features.to(device=points.device, dtype=points.dtype)
            )
            feature_tokens = feature_tokens[:, None].expand(
                -1, self.n_future_frames + 1, -1, -1
            )
            points = self.utonia_feature_projection(
                torch.cat((points, feature_tokens), dim=-1)
            )
        point_positions = self.position_embedding[:, 2:].to(
            device=points.device, dtype=points.dtype
        )
        points = points + point_positions.reshape(
            1, self.n_future_frames + 1, self.n_points, self.latent_dim
        )
        controls = torch.stack(
            (
                self.linear_velocity_encoder(initial_linear_velocity.squeeze(1)),
                self.angular_velocity_encoder(initial_angular_velocity.squeeze(1)),
            ),
            dim=1,
        )
        controls = controls[:, None].expand(-1, self.n_future_frames + 1, -1, -1)
        temb = self.time_embedding(frame_times).to(dtype=points.dtype)
        return points, controls, temb

    def decode_states(
        self, points: torch.Tensor, temb: torch.Tensor, init_pc: torch.Tensor
    ) -> torch.Tensor:
        """Decode a post-block point state into the model's legacy output convention."""
        offset = self.output_head(points[:, 1:], temb[:, 1:]).unsqueeze(2)
        return offset if self.objective_type == "flow" else offset + init_pc.unsqueeze(1)
