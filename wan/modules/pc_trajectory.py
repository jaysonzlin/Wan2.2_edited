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
    """Predict point-cloud futures from velocity or clean-history conditions."""

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
        conditioning: str = "velocity",
        history_frames: int = 1,
    ):
        super().__init__()
        if objective_type not in {"flow", "ddpm"}:
            raise ValueError("objective_type must be 'flow' or 'ddpm'")
        if conditioning not in {"velocity", "history"}:
            raise ValueError("conditioning must be 'velocity' or 'history'")
        if conditioning == "history" and (
            not isinstance(history_frames, int)
            or isinstance(history_frames, bool)
            or history_frames != 4
        ):
            raise ValueError("history_frames must be 4 when conditioning is 'history'")
        if conditioning == "history" and objective_type != "ddpm":
            raise ValueError("history conditioning requires objective_type 'ddpm'")
        if conditioning == "history" and n_future_frames != 45:
            raise ValueError("n_future_frames must be 45 when conditioning is 'history'")
        if conditioning == "velocity" and history_frames != 1:
            raise ValueError("history_frames must be 1 when conditioning is 'velocity'")
        if latent_dim % 64:
            raise ValueError("latent_dim must be divisible by 64")
        if num_heads != latent_dim // 64:
            raise ValueError("num_heads must equal latent_dim // 64")
        if not point_embed:
            raise ValueError("point_embed must be true")
        if utonia_feature_dim is not None and utonia_feature_dim <= 0:
            raise ValueError("utonia_feature_dim must be positive")

        self.objective_type = objective_type
        self.conditioning = conditioning
        self.history_frames = history_frames
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
        if conditioning == "velocity":
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
            physctrl_position_embedding(
                n_points, self.history_frames + self.n_future_frames, latent_dim
            ),
            persistent=False,
        )

    def forward(
        self,
        noisy_future_state: torch.Tensor,
        frame_times: torch.Tensor,
        init_pc: torch.Tensor,
        initial_linear_velocity: torch.Tensor | None,
        initial_angular_velocity: torch.Tensor | None,
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
        initial_linear_velocity: torch.Tensor | None,
        initial_angular_velocity: torch.Tensor | None,
        utonia_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Encode a trajectory into point/control states for externally interleaved blocks."""
        batch = noisy_future_state.shape[0]
        expected = (batch, self.n_future_frames, 1, self.n_points, 3)
        if noisy_future_state.shape != expected:
            raise ValueError(f"noisy_future_state must have shape {expected}")
        expected_known_shape = (
            (batch, 1, self.n_points, 3)
            if self.conditioning == "velocity"
            else (batch, self.history_frames, 1, self.n_points, 3)
        )
        if init_pc.shape != expected_known_shape:
            raise ValueError(f"init_pc must have shape {expected_known_shape}")
        if frame_times.shape != (batch, self.history_frames + self.n_future_frames):
            raise ValueError("frame_times must have shape (B, 49)")
        if self.conditioning == "history" and not torch.equal(
            frame_times[:, : self.history_frames],
            torch.zeros_like(frame_times[:, : self.history_frames]),
        ):
            raise ValueError("known history frame times must be zero")
        if self.objective_type == "flow" and not torch.equal(
            frame_times[:, 0], torch.zeros_like(frame_times[:, 0])
        ):
            raise ValueError("frame_times[:, 0] must be zero")
        if self.conditioning == "velocity" and (
            initial_linear_velocity is None
            or initial_angular_velocity is None
            or initial_linear_velocity.shape != (batch, 1, 3)
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

        known_positions = (
            init_pc.unsqueeze(1) if self.conditioning == "velocity" else init_pc
        )
        source_position = known_positions[:, :1]
        future_positions = (
            source_position + noisy_future_state
            if self.objective_type == "flow"
            else noisy_future_state
        )
        coordinates = torch.cat((known_positions, future_positions), dim=1).squeeze(2)
        points = self.input_encoder(coordinates.reshape(-1, self.n_points, 3))
        points = points.reshape(
            batch,
            self.history_frames + self.n_future_frames,
            self.n_points,
            self.latent_dim,
        )
        if self.utonia_feature_dim is not None:
            assert utonia_features is not None
            feature_tokens = self.utonia_feature_norm(
                utonia_features.to(device=points.device, dtype=points.dtype)
            )
            feature_tokens = feature_tokens[:, None].expand(
                -1, self.history_frames + self.n_future_frames, -1, -1
            )
            points = self.utonia_feature_projection(
                torch.cat((points, feature_tokens), dim=-1)
            )
        point_positions = self.position_embedding[:, 2:].to(
            device=points.device, dtype=points.dtype
        )
        points = points + point_positions.reshape(
            1,
            self.history_frames + self.n_future_frames,
            self.n_points,
            self.latent_dim,
        )
        if self.conditioning == "velocity":
            assert initial_linear_velocity is not None
            assert initial_angular_velocity is not None
            controls = torch.stack(
                (
                    self.linear_velocity_encoder(initial_linear_velocity.squeeze(1)),
                    self.angular_velocity_encoder(initial_angular_velocity.squeeze(1)),
                ),
                dim=1,
            )
            controls = controls[:, None].expand(
                -1, self.history_frames + self.n_future_frames, -1, -1
            )
        else:
            controls = None
        temb = self.time_embedding(frame_times).to(dtype=points.dtype)
        return points, controls, temb

    def decode_states(
        self, points: torch.Tensor, temb: torch.Tensor, init_pc: torch.Tensor
    ) -> torch.Tensor:
        """Decode a post-block point state into the model's legacy output convention."""
        offset = self.output_head(
            points[:, self.history_frames :], temb[:, self.history_frames :]
        ).unsqueeze(2)
        if self.objective_type == "flow":
            return offset
        anchor = init_pc.unsqueeze(1) if self.conditioning == "velocity" else init_pc[:, :1]
        return offset + anchor
