"""Aligned DDPM batches and per-object objectives for joint Wan--PhysCtrl training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MultiObjectPCDDPMBatch:
    """Noised future trajectories for every object in a batch-size-one video clip."""

    model_input: torch.Tensor
    target: torch.Tensor
    frame_times: torch.Tensor
    timesteps: torch.Tensor


def make_aligned_multi_object_pc_ddpm_batch(
    point_clouds: torch.Tensor, scheduler, generator: torch.Generator
) -> MultiObjectPCDDPMBatch:
    """Noise frames 1--48 with one DDPM timestep per video and independent object noise.

    ``point_clouds`` has shape ``[batch, objects, 49, 1, points, xyz]``. The same sampled
    discrete timestep is expanded to all objects from the same video, whereas the Gaussian noise
    remains independently sampled at every element.
    """
    if point_clouds.ndim != 6 or point_clouds.shape[2] != 49:
        raise ValueError("point_clouds must have shape [B, K, 49, 1, N, 3]")
    batch_size, object_count = point_clouds.shape[:2]
    future_points = point_clouds[:, :, 1:]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=point_clouds.device,
        generator=generator,
    )
    noise = torch.randn(
        future_points.shape,
        device=future_points.device,
        dtype=future_points.dtype,
        generator=generator,
    )
    flat_future = future_points.flatten(0, 1)
    flat_noise = noise.flatten(0, 1)
    flat_timesteps = timesteps[:, None].expand(batch_size, object_count).reshape(-1)
    noised = scheduler.add_noise(flat_future, flat_noise, flat_timesteps).unflatten(
        0, (batch_size, object_count)
    )
    frame_times = (
        timesteps[:, None, None]
        .expand(batch_size, object_count, 49)
        .to(point_clouds.dtype)
    )
    return MultiObjectPCDDPMBatch(
        model_input=noised,
        target=future_points,
        frame_times=frame_times,
        timesteps=timesteps,
    )


def per_object_pc_x0_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return individual object x0 losses and their unaveraged total."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim < 3:
        raise ValueError("prediction must include batch and object dimensions")
    losses = (prediction - target).square().flatten(start_dim=2).mean(dim=2)
    return losses, losses.sum()


def per_object_rigid_edge_length_loss(
    initial_point_clouds: torch.Tensor,
    prediction: torch.Tensor,
    neighbors: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return each object's scale-normalized drift from its frame-zero geometry."""
    if (
        initial_point_clouds.ndim != 5
        or initial_point_clouds.shape[2] != 1
        or initial_point_clouds.shape[-1] != 3
    ):
        raise ValueError("initial_point_clouds must have shape [B, K, 1, N, 3]")
    if prediction.ndim != 6 or prediction.shape[3] != 1 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [B, K, T, 1, N, 3]")
    if (
        prediction.shape[:2] != initial_point_clouds.shape[:2]
        or prediction.shape[-2:] != initial_point_clouds.shape[-2:]
    ):
        raise ValueError(
            "initial_point_clouds and prediction must agree on batch, object, point, and coordinate dimensions"
        )
    point_count = initial_point_clouds.shape[-2]
    if (
        not isinstance(neighbors, int)
        or isinstance(neighbors, bool)
        or not 1 <= neighbors < point_count
    ):
        raise ValueError("neighbors must be an integer in [1, point_count)")

    initial = initial_point_clouds.squeeze(2)
    future = prediction.squeeze(3)
    batch_size, object_count, frame_count, _, _ = future.shape
    neighbor_indices = torch.topk(
        torch.cdist(initial.float(), initial.float()),
        k=neighbors + 1,
        largest=False,
    ).indices[..., 1:]

    initial_neighbors = torch.gather(
        initial.unsqueeze(3).expand(-1, -1, -1, neighbors, -1),
        dim=2,
        index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 3),
    )
    initial_lengths = torch.linalg.vector_norm(
        initial_neighbors - initial.unsqueeze(3), dim=-1
    )

    flat_future = future.reshape(
        batch_size * object_count * frame_count, point_count, 3
    )
    flat_indices = (
        neighbor_indices.unsqueeze(2)
        .expand(-1, -1, frame_count, -1, -1)
        .reshape(batch_size * object_count * frame_count, point_count, neighbors)
    )
    future_neighbors = torch.gather(
        flat_future.unsqueeze(2).expand(-1, -1, neighbors, -1),
        dim=1,
        index=flat_indices.unsqueeze(-1).expand(-1, -1, -1, 3),
    )
    future_lengths = torch.linalg.vector_norm(
        future_neighbors - flat_future.unsqueeze(2), dim=-1
    ).reshape(batch_size, object_count, frame_count, point_count, neighbors)

    edge_drift = (future_lengths - initial_lengths.unsqueeze(2)).square() / (
        initial_lengths.unsqueeze(2).square() + epsilon
    )
    return edge_drift.mean(dim=(2, 3, 4))


def _physctrl_deformation_frame_loss(
    particle_x: torch.Tensor,
    particle_x_two_ahead: torch.Tensor,
    particle_f: torch.Tensor,
    particle_f_next: torch.Tensor,
    particle_c: torch.Tensor,
    particle_volume: torch.Tensor,
    baseline: torch.Tensor,
    *,
    dt: float,
    grid_size: int,
    grid_lim: float,
) -> torch.Tensor:
    """Evaluate one baseline-corrected quadratic-B-spline P2G/G2P transition."""
    batch_size, object_count, point_count, _ = particle_x.shape
    transfer_count = batch_size * object_count
    dx = grid_lim / grid_size
    inv_dx = 1.0 / dx
    positions = particle_x.float().flatten(0, 1)
    velocities = (
        ((particle_x_two_ahead - particle_x) / (2.0 * dt)).float().flatten(0, 1)
    )
    deformation = particle_f.float().flatten(0, 1)
    deformation_next = particle_f_next.float().flatten(0, 1)
    affine_velocity = particle_c.float().flatten(0, 1)
    volumes = particle_volume.float().flatten(0, 1)
    expected_baseline = baseline.float().flatten(0, 1)

    grid_positions = positions * inv_dx
    base = (grid_positions - 0.5).to(torch.int64)
    fractions = grid_positions - base
    if bool((base < 0).any()) or bool((base + 2 >= grid_size).any()):
        raise ValueError(
            "deformation grid transform places a particle outside the transfer domain"
        )
    weights = torch.stack(
        (
            0.5 * (1.5 - fractions).square(),
            0.75 - (fractions - 1.0).square(),
            0.5 * (fractions - 0.5).square(),
        ),
        dim=-1,
    )
    derivatives = torch.stack(
        (fractions - 1.5, -2.0 * (fractions - 1.0), fractions - 0.5), dim=-1
    )
    cell_count = grid_size**3
    grid_mass = torch.zeros(
        (transfer_count, cell_count), device=positions.device, dtype=positions.dtype
    )
    grid_velocity = torch.zeros(
        (transfer_count, cell_count, 3), device=positions.device, dtype=positions.dtype
    )

    for i in range(3):
        for j in range(3):
            for k in range(3):
                offset = positions.new_tensor((i, j, k))
                flat_indices = (
                    (base[:, :, 0] + i) * grid_size * grid_size
                    + (base[:, :, 1] + j) * grid_size
                    + base[:, :, 2]
                    + k
                )
                weight = weights[:, :, 0, i] * weights[:, :, 1, j] * weights[:, :, 2, k]
                dpos = (offset - fractions) * dx
                affine_term = torch.matmul(affine_velocity, dpos.unsqueeze(-1)).squeeze(
                    -1
                )
                grid_mass = grid_mass.scatter_add(1, flat_indices, weight * volumes)
                grid_velocity = grid_velocity.scatter_add(
                    1,
                    flat_indices.unsqueeze(-1).expand(-1, -1, 3),
                    weight.unsqueeze(-1)
                    * volumes.unsqueeze(-1)
                    * (velocities + affine_term),
                )
    grid_velocity = grid_velocity / torch.where(
        grid_mass > 1e-15, grid_mass, torch.ones_like(grid_mass)
    ).unsqueeze(-1)

    velocity_gradient = torch.zeros_like(deformation)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                flat_indices = (
                    (base[:, :, 0] + i) * grid_size * grid_size
                    + (base[:, :, 1] + j) * grid_size
                    + base[:, :, 2]
                    + k
                )
                dweight = (
                    torch.stack(
                        (
                            derivatives[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * derivatives[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * derivatives[:, :, 2, k],
                        ),
                        dim=-1,
                    )
                    * inv_dx
                )
                local_velocity = grid_velocity.gather(
                    1, flat_indices.unsqueeze(-1).expand(-1, -1, 3)
                )
                velocity_gradient = velocity_gradient + local_velocity.unsqueeze(
                    -1
                ) * dweight.unsqueeze(-2)
    identity = torch.eye(3, device=positions.device, dtype=positions.dtype).expand_as(
        deformation
    )
    predicted_deformation = (identity + velocity_gradient * dt) @ deformation
    raw_residual = (predicted_deformation - deformation_next).abs()
    excess = (raw_residual - expected_baseline).clamp_min(0)
    penalty = functional.smooth_l1_loss(
        excess, torch.zeros_like(excess), beta=0.01, reduction="none"
    )
    return penalty.mean(dim=(1, 2, 3)).unflatten(0, (batch_size, object_count))


def per_object_baseline_corrected_deform_loss(
    initial_point_clouds: torch.Tensor,
    prediction: torch.Tensor,
    deform_f: torch.Tensor,
    deform_c: torch.Tensor,
    deform_volume: torch.Tensor,
    deform_baseline: torch.Tensor,
    deform_grid_origin: torch.Tensor,
    deform_grid_scale: torch.Tensor,
    *,
    dt: float = 0.02,
    grid_size: int = 125,
    grid_lim: float = 10.0,
) -> torch.Tensor:
    """Return per-object loss above the GT approximation residual of the PhysCtrl update.

    Saved MLS fields are treated as frozen supervision.  The x0 prediction supplies positions,
    while the saved baseline removes irreducible error from reconstructing MPM F and C from
    positions alone.
    """
    if (
        initial_point_clouds.ndim != 5
        or initial_point_clouds.shape[2] != 1
        or initial_point_clouds.shape[-1] != 3
    ):
        raise ValueError("initial_point_clouds must have shape [B, K, 1, N, 3]")
    if prediction.ndim != 6 or prediction.shape[3] != 1 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [B, K, T-1, 1, N, 3]")
    batch_size, object_count, _, point_count, _ = initial_point_clouds.shape
    frame_count = prediction.shape[2] + 1
    if (
        prediction.shape[:2] != (batch_size, object_count)
        or prediction.shape[-2] != point_count
    ):
        raise ValueError(
            "initial_point_clouds and prediction must agree on batch, object, and point dimensions"
        )
    field_shape = (batch_size, object_count, frame_count, 1, point_count, 3, 3)
    if deform_f.shape != field_shape or deform_c.shape != field_shape:
        raise ValueError("deform_f and deform_c must have shape [B, K, T, 1, N, 3, 3]")
    if deform_volume.shape != (batch_size, object_count, 1, point_count):
        raise ValueError("deform_volume must have shape [B, K, 1, N]")
    if deform_baseline.shape != (
        batch_size,
        object_count,
        frame_count - 2,
        1,
        point_count,
        3,
        3,
    ):
        raise ValueError("deform_baseline must have shape [B, K, T-2, 1, N, 3, 3]")
    if deform_grid_origin.shape != (batch_size, object_count, 3):
        raise ValueError("deform_grid_origin must have shape [B, K, 3]")
    if deform_grid_scale.shape != (batch_size, object_count, 1):
        raise ValueError("deform_grid_scale must have shape [B, K, 1]")
    if frame_count < 3 or dt <= 0 or grid_size < 5 or grid_lim <= 0:
        raise ValueError("invalid deformation objective settings")

    world_positions = torch.cat(
        (initial_point_clouds.unsqueeze(2), prediction), dim=2
    ).squeeze(3)
    grid_positions = (
        world_positions - deform_grid_origin[:, :, None, None, :]
    ) * deform_grid_scale[:, :, None, None, :] + 2.0 * (grid_lim / grid_size)
    deformation = deform_f.squeeze(3)
    affine_velocity = deform_c.squeeze(3)
    baseline = deform_baseline.squeeze(3)
    losses = torch.zeros(
        (batch_size, object_count), device=prediction.device, dtype=torch.float32
    )
    for frame in range(frame_count - 2):
        arguments = (
            grid_positions[:, :, frame],
            grid_positions[:, :, frame + 2],
            deformation[:, :, frame],
            deformation[:, :, frame + 1],
            affine_velocity[:, :, frame],
            deform_volume.squeeze(2),
            baseline[:, :, frame],
        )
        if prediction.requires_grad:
            frame_loss = checkpoint(
                lambda *values: _physctrl_deformation_frame_loss(
                    *values, dt=dt, grid_size=grid_size, grid_lim=grid_lim
                ),
                *arguments,
                use_reentrant=False,
            )
        else:
            frame_loss = _physctrl_deformation_frame_loss(
                *arguments, dt=dt, grid_size=grid_size, grid_lim=grid_lim
            )
        losses = losses + frame_loss
    return losses / (frame_count - 2)
