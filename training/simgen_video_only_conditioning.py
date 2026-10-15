"""Synthetic non-video conditioning for the SimGen video-only ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SYNTHETIC_OBJECT_SLOTS = 3
HISTORY_FRAMES = 4
FUTURE_FRAMES = 45
PC_GENERATOR_SEED_OFFSET = 1


@dataclass(frozen=True)
class SyntheticPCConditioning:
    """Shape-compatible random inputs for the retained PC pathway."""

    init_pc: torch.Tensor
    noisy_future_state: torch.Tensor
    utonia_features: torch.Tensor
    frame_times: torch.Tensor


class GeneratorStateBundle:
    """Make named ``torch.Generator`` states checkpointable by Accelerate."""

    def __init__(self, **generators: torch.Generator) -> None:
        if not generators or any(
            not isinstance(generator, torch.Generator)
            for generator in generators.values()
        ):
            raise ValueError("GeneratorStateBundle requires named torch.Generator instances")
        self._generators = generators

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: generator.get_state().clone()
            for name, generator in self._generators.items()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self._generators):
            raise ValueError("generator checkpoint names do not match this run")
        for name, generator in self._generators.items():
            generator.set_state(state[name])


def make_video_only_generators(
    device: torch.device | str, seed: int
) -> tuple[torch.Generator, torch.Generator]:
    """Return independent, repeatable video and synthetic-PC RNG streams."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    return (
        torch.Generator(device=device).manual_seed(seed),
        torch.Generator(device=device).manual_seed(seed + PC_GENERATOR_SEED_OFFSET),
    )


def _validate_dimensions(
    *, batch_size: int, point_count: int, utonia_feature_dim: int,
    num_train_timesteps: int | None = None,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("point_count", point_count),
        ("utonia_feature_dim", utonia_feature_dim),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_train_timesteps is not None and (
        not isinstance(num_train_timesteps, int)
        or isinstance(num_train_timesteps, bool)
        or num_train_timesteps <= 0
    ):
        raise ValueError("num_train_timesteps must be a positive integer")


def make_training_conditioning(
    *,
    batch_size: int,
    point_count: int,
    utonia_feature_dim: int,
    num_train_timesteps: int,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> SyntheticPCConditioning:
    """Draw fresh independent noise for one training or scalar-validation call."""
    _validate_dimensions(
        batch_size=batch_size,
        point_count=point_count,
        utonia_feature_dim=utonia_feature_dim,
        num_train_timesteps=num_train_timesteps,
    )
    shape_prefix = (batch_size, SYNTHETIC_OBJECT_SLOTS)
    init_pc = torch.randn(
        (*shape_prefix, HISTORY_FRAMES, 1, point_count, 3),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    noisy_future_state = torch.randn(
        (*shape_prefix, FUTURE_FRAMES, 1, point_count, 3),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    utonia_features = torch.randn(
        (*shape_prefix, point_count, utonia_feature_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    future_times = torch.randint(
        num_train_timesteps,
        shape_prefix,
        device=device,
        generator=generator,
    ).to(dtype=dtype).unsqueeze(-1).expand(-1, -1, FUTURE_FRAMES)
    history_times = torch.zeros(
        (*shape_prefix, HISTORY_FRAMES), device=device, dtype=dtype
    )
    return SyntheticPCConditioning(
        init_pc=init_pc,
        noisy_future_state=noisy_future_state,
        utonia_features=utonia_features,
        frame_times=torch.cat((history_times, future_times), dim=-1),
    )


def make_sampling_conditioning(
    *,
    point_count: int,
    utonia_feature_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the fixed history/features used through one complete sampler run."""
    _validate_dimensions(
        batch_size=1,
        point_count=point_count,
        utonia_feature_dim=utonia_feature_dim,
    )
    history = torch.randn(
        (SYNTHETIC_OBJECT_SLOTS, HISTORY_FRAMES, point_count, 3),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    features = torch.randn(
        (SYNTHETIC_OBJECT_SLOTS, point_count, utonia_feature_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return history, features
