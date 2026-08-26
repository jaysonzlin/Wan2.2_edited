"""Full-token bidirectional coupling between Wan video and PhysCtrl trajectories."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import sinusoidal_embedding_1d


class BidirectionalWanPhysCtrlBridge(nn.Module):
    """Exchange video and per-object point tokens through a shared attention width.

    Video queries attend to the concatenation of every object's point tokens. Point queries are
    evaluated independently per object against the single video token sequence, so this module
    introduces no direct object-to-object attention.
    """

    def __init__(
        self, video_dim: int = 3072, point_dim: int = 256, interaction_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if interaction_dim % num_heads:
            raise ValueError("interaction_dim must be divisible by num_heads")
        self.video_dim = video_dim
        self.point_dim = point_dim
        self.interaction_dim = interaction_dim
        self.num_heads = num_heads
        self.head_dim = interaction_dim // num_heads

        self.video_norm = nn.LayerNorm(video_dim)
        self.point_norm = nn.LayerNorm(point_dim)
        self.video_to_points_q = nn.Linear(video_dim, interaction_dim)
        self.video_to_points_k = nn.Linear(point_dim, interaction_dim)
        self.video_to_points_v = nn.Linear(point_dim, interaction_dim)
        self.video_to_points_out = nn.Linear(interaction_dim, video_dim)
        self.points_to_video_q = nn.Linear(point_dim, interaction_dim)
        self.points_to_video_k = nn.Linear(video_dim, interaction_dim)
        self.points_to_video_v = nn.Linear(video_dim, interaction_dim)
        self.points_to_video_out = nn.Linear(interaction_dim, point_dim)
        self.video_gate = nn.Parameter(torch.zeros(video_dim))
        self.point_gate = nn.Parameter(torch.zeros(point_dim))
        self._initialize_projections()

    def _initialize_projections(self) -> None:
        """Match FantasyWorld's bridge initialization for the attention projections."""
        for projection in (
            self.video_to_points_q,
            self.video_to_points_k,
            self.video_to_points_v,
            self.video_to_points_out,
            self.points_to_video_q,
            self.points_to_video_k,
            self.points_to_video_v,
            self.points_to_video_out,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch, query_length = query.shape[:2]
        key_length = key.shape[1]
        query = query.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value)
        return output.transpose(1, 2).reshape(batch, query_length, self.interaction_dim)

    def forward(
        self, video_tokens: torch.Tensor, point_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual-updated ``[B,Lv,Dv]`` and ``[B,K,Lp,Dp]`` states."""
        if video_tokens.ndim != 3 or point_tokens.ndim != 4:
            raise ValueError("video_tokens must be [B, Lv, Dv] and point_tokens [B, K, Lp, Dp]")
        batch, object_count, point_length, point_dim = point_tokens.shape
        if video_tokens.shape[0] != batch or video_tokens.shape[-1] != self.video_dim:
            raise ValueError("video token shape does not match bridge dimensions")
        if object_count == 0 or point_dim != self.point_dim:
            raise ValueError("point token shape does not match bridge dimensions")

        normalized_video = self.video_norm(video_tokens)
        normalized_points = self.point_norm(point_tokens)
        flat_points = normalized_points.reshape(batch, object_count * point_length, point_dim)
        video_update = self.video_to_points_out(
            self._attention(
                self.video_to_points_q(normalized_video),
                self.video_to_points_k(flat_points),
                self.video_to_points_v(flat_points),
            )
        )

        flat_point_queries = self.points_to_video_q(normalized_points).flatten(0, 1)
        repeated_video = normalized_video[:, None].expand(-1, object_count, -1, -1).flatten(0, 1)
        point_update = self.points_to_video_out(
            self._attention(
                flat_point_queries,
                self.points_to_video_k(repeated_video),
                self.points_to_video_v(repeated_video),
            )
        ).unflatten(0, (batch, object_count))
        return (
            video_tokens + self.video_gate * video_update,
            point_tokens + self.point_gate * point_update,
        )


class JointWanPhysCtrlModel(nn.Module):
    """Run Wan's final eight blocks jointly with one PhysCtrl branch per object."""

    def __init__(self, wan_model: nn.Module, pc_model: nn.Module) -> None:
        super().__init__()
        if len(wan_model.blocks) < 8:
            raise ValueError("Wan model must contain at least 8 transformer blocks")
        if len(pc_model.blocks) != 8:
            raise ValueError("PhysCtrl model must contain exactly 8 transformer blocks")
        self.wan_model = wan_model
        self.pc_model = pc_model
        self.paired_start = len(wan_model.blocks) - 8
        self.bridges = nn.ModuleList(
            BidirectionalWanPhysCtrlBridge(
                video_dim=wan_model.dim,
                point_dim=pc_model.latent_dim,
                interaction_dim=512,
                num_heads=8,
            )
            for _ in range(8)
        )

    def _prepare_wan_states(self, x, t, context, seq_len, y):
        wan = self.wan_model
        if getattr(wan, "model_type", None) == "i2v" and y is None:
            raise ValueError("I2V Wan models require a conditioning input")
        device = wan.patch_embedding.weight.device
        if wan.freqs.device != device:
            wan.freqs = wan.freqs.to(device)
        if y is not None:
            x = [torch.cat((video, condition), dim=0) for video, condition in zip(x, y)]
        embedded = [wan.patch_embedding(video.unsqueeze(0)) for video in x]
        grid_sizes = torch.stack([torch.tensor(video.shape[2:], dtype=torch.long) for video in embedded])
        tokens = [video.flatten(2).transpose(1, 2) for video in embedded]
        seq_lens = torch.tensor([video.size(1) for video in tokens], dtype=torch.long)
        if seq_lens.max() > seq_len:
            raise ValueError("seq_len is shorter than the Wan patch token sequence")
        hidden = torch.cat(
            [torch.cat((video, video.new_zeros(1, seq_len - video.size(1), video.size(2))), dim=1)
             for video in tokens]
        )
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            batch_size = t.size(0)
            timestep_embedding = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, t.flatten()).unflatten(0, (batch_size, seq_len)).float()
            )
            modulation = wan.time_projection(timestep_embedding).unflatten(2, (6, wan.dim))
        encoded_context = wan.text_embedding(
            torch.stack(
                [torch.cat((item, item.new_zeros(wan.text_len - item.size(0), item.size(1)))) for item in context]
            )
        )
        return hidden, timestep_embedding, modulation, {
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": wan.freqs,
            "context": encoded_context,
            "context_lens": None,
        }

    def _run_paired_block(self, wan_block, pc_block, bridge, hidden, point_states, controls, pc_temb, modulation, wan_kwargs, batch_size, object_count):
        terms = wan_block.modulation_terms(modulation)
        hidden = wan_block.run_self_attention(
            hidden, terms, wan_kwargs["seq_lens"], wan_kwargs["grid_sizes"], wan_kwargs["freqs"]
        )
        hidden = wan_block.run_text_cross_attention(
            hidden, wan_kwargs["context"], wan_kwargs["context_lens"]
        )
        point_states, controls = pc_block(point_states, controls, pc_temb)
        point_tokens = point_states.reshape(
            batch_size, object_count, point_states.shape[1] * point_states.shape[2], point_states.shape[3]
        )
        hidden, point_tokens = bridge(hidden, point_tokens)
        point_states = point_tokens.reshape_as(point_states)
        return wan_block.run_mlp(hidden, terms), point_states, controls

    def forward(
        self,
        video_x: list[torch.Tensor],
        video_t: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        noisy_future_state: torch.Tensor,
        frame_times: torch.Tensor,
        init_pc: torch.Tensor,
        initial_linear_velocity: torch.Tensor | None = None,
        initial_angular_velocity: torch.Tensor | None = None,
        utonia_features: torch.Tensor | None = None,
        video_condition: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return Wan velocities and DDPM x0 predictions for every object in the clip."""
        if len(video_x) != 1 or noisy_future_state.shape[0] != 1:
            raise ValueError("JointWanPhysCtrlModel currently requires train_batch_size=1")
        if noisy_future_state.ndim != 6:
            raise ValueError("noisy_future_state must have shape [B, K, T, 1, N, 3]")
        batch_size, object_count = noisy_future_state.shape[:2]
        if (
            frame_times.ndim != 3
            or frame_times.shape[:2] != (batch_size, object_count)
            or frame_times.shape[2]
            != noisy_future_state.shape[2] + self.pc_model.history_frames
        ):
            raise ValueError(
                "frame_times must include every known PC history slot"
            )
        hidden, timestep_embedding, modulation, wan_kwargs = self._prepare_wan_states(
            video_x, video_t, context, seq_len, video_condition
        )
        flat = lambda tensor: tensor.flatten(0, 1)
        point_states, controls, pc_temb = self.pc_model.encode_states(
            flat(noisy_future_state),
            flat(frame_times),
            flat(init_pc),
            None if initial_linear_velocity is None else flat(initial_linear_velocity),
            None if initial_angular_velocity is None else flat(initial_angular_velocity),
            None if utonia_features is None else flat(utonia_features),
        )
        for block in self.wan_model.blocks[:self.paired_start]:
            if self.training and getattr(self.wan_model, "gradient_checkpointing", False):
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda states, module=block: module(states, e=modulation, **wan_kwargs),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, e=modulation, **wan_kwargs)
        for wan_block, pc_block, bridge in (
            zip(self.wan_model.blocks[self.paired_start:], self.pc_model.blocks, self.bridges)
        ):
            run_block = lambda hidden, points, controls, wan_block=wan_block, pc_block=pc_block, bridge=bridge: self._run_paired_block(
                wan_block, pc_block, bridge, hidden, points, controls, pc_temb,
                modulation, wan_kwargs, batch_size, object_count,
            )
            if self.training and getattr(self.wan_model, "gradient_checkpointing", False):
                hidden, point_states, controls = torch.utils.checkpoint.checkpoint(
                    run_block, hidden, point_states, controls, use_reentrant=False
                )
            else:
                hidden, point_states, controls = run_block(hidden, point_states, controls)
        video_output = self.wan_model.unpatchify(self.wan_model.head(hidden, timestep_embedding), wan_kwargs["grid_sizes"])
        pc_output = self.pc_model.decode_states(point_states, pc_temb, flat(init_pc)).unflatten(0, (batch_size, object_count))
        return [item.float() for item in video_output], pc_output
