"""Derive approximate PhysCtrl deformation fields for a TD point-cloud sample.

The source TD trajectories contain positions only.  This module reconstructs local affine
deformation and velocity fields from a frame-zero neighbourhood graph, then saves the
unavoidable GT P2G/G2P update error as a per-entry baseline for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

DEFORM_DT = 0.02
DEFORM_GRID_LIM = 10.0
DEFORM_GRID_SIZE = 125


def _object_paths(sample_dir: Path) -> list[Path]:
    paths = sorted((sample_dir / "objects").glob("*/pc.hdf5"))
    if not paths:
        raise ValueError(f"{sample_dir}: no object pc.hdf5 files found")
    return paths


def _load_trajectories(sample_dir: Path) -> list[tuple[Path, np.ndarray]]:
    result = []
    for path in _object_paths(sample_dir):
        with h5py.File(path, "r") as source:
            if "point_cloud" not in source:
                raise KeyError(f"{path}: missing point_cloud")
            trajectory = np.asarray(source["point_cloud"][:], dtype=np.float64)
        if (
            trajectory.ndim != 4
            or trajectory.shape[1] != 1
            or trajectory.shape[-1] != 3
        ):
            raise ValueError(f"{path}: point_cloud must have shape [T, 1, N, 3]")
        if trajectory.shape[0] < 3:
            raise ValueError(f"{path}: at least three frames are required")
        if not np.isfinite(trajectory).all():
            raise ValueError(f"{path}: point_cloud contains non-finite values")
        result.append((path, trajectory[:, 0]))
    return result


def make_shared_grid_transform(
    trajectories: list[np.ndarray], *, grid_size: int, grid_lim: float = DEFORM_GRID_LIM
) -> tuple[np.ndarray, float]:
    """Return a common world-to-grid-world transform with a two-cell safety border."""
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    all_positions = np.concatenate(
        [trajectory.reshape(-1, 3) for trajectory in trajectories]
    )
    origin = all_positions.min(axis=0)
    extent = float((all_positions.max(axis=0) - origin).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("all trajectories must have non-zero finite spatial extent")
    dx = grid_lim / grid_size
    scale = (grid_lim - 4.0 * dx) / extent
    return origin, float(scale)


def world_to_grid(
    trajectory: np.ndarray,
    origin: np.ndarray,
    scale: float,
    *,
    grid_size: int,
    grid_lim: float = DEFORM_GRID_LIM,
) -> np.ndarray:
    dx = grid_lim / grid_size
    return (trajectory - origin) * scale + 2.0 * dx


def _frame_zero_knn(
    points: np.ndarray, neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    point_count = points.shape[0]
    if (
        not isinstance(neighbors, int)
        or isinstance(neighbors, bool)
        or not 1 <= neighbors < point_count
    ):
        raise ValueError("neighbors must be an integer in [1, point_count)")
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    indices = np.argpartition(distances, kth=neighbors, axis=1)[:, 1 : neighbors + 1]
    neighbor_distances = np.take_along_axis(distances, indices, axis=1)
    return indices, neighbor_distances


def _affine_fit(
    source: np.ndarray, target: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Fit target-neighbour displacements = A @ source-neighbour displacements."""
    source_delta = source[indices] - source[:, None]
    target_delta = target[indices] - target[:, None]
    normal = np.einsum("nk,nki,nkj->nij", weights, source_delta, source_delta)
    trace = np.trace(normal, axis1=-2, axis2=-1)
    normal += np.eye(3)[None] * np.maximum(trace, 1.0)[:, None, None] * 1e-7
    rhs = np.einsum("nk,nki,nkj->nij", weights, target_delta, source_delta)
    return np.linalg.solve(normal, rhs.transpose(0, 2, 1)).transpose(0, 2, 1)


def estimate_mls_fields(
    positions: np.ndarray, *, neighbors: int, dt: float = DEFORM_DT
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F from rest displacements and C from finite-difference velocities."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T, N, 3]")
    frame_count, point_count, _ = positions.shape
    indices, distances = _frame_zero_knn(positions[0], neighbors)
    sigma_sq = np.maximum(np.square(distances).mean(axis=1, keepdims=True), 1e-12)
    weights = np.exp(-np.square(distances) / sigma_sq)

    deformation = np.empty((frame_count, point_count, 3, 3), dtype=np.float64)
    for frame in range(frame_count):
        deformation[frame] = _affine_fit(
            positions[0], positions[frame], indices, weights
        )

    velocity = np.empty_like(positions)
    velocity[0] = (positions[1] - positions[0]) / dt
    velocity[-1] = (positions[-1] - positions[-2]) / dt
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    affine_velocity = np.empty_like(deformation)
    for frame in range(frame_count):
        affine_velocity[frame] = _affine_fit(
            positions[frame], velocity[frame], indices, weights
        )
    return deformation.astype(np.float32), affine_velocity.astype(np.float32)


def physctrl_update_residual(
    positions: np.ndarray,
    deformation: np.ndarray,
    affine_velocity: np.ndarray,
    volume: np.ndarray,
    *,
    dt: float = DEFORM_DT,
    grid_size: int = DEFORM_GRID_SIZE,
    grid_lim: float = DEFORM_GRID_LIM,
) -> np.ndarray:
    """Return PhysCtrl-style absolute F-update residuals for transitions 0..T-3.

    This deliberately mirrors PhysCtrl's quadratic B-spline P2G/G2P transfer.  It runs a
    transition at a time so preprocessing and the training implementation have bounded grid
    memory rather than allocating one grid for every frame.
    """
    if positions.ndim != 3 or deformation.shape != positions.shape[:2] + (3, 3):
        raise ValueError("positions and deformation have incompatible shapes")
    if affine_velocity.shape != deformation.shape:
        raise ValueError("affine_velocity must have the same shape as deformation")
    if volume.shape != positions.shape[1:2]:
        raise ValueError("volume must have shape [N]")
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    dx = grid_lim / grid_size
    inv_dx = 1.0 / dx
    frame_count, point_count, _ = positions.shape
    residuals = np.empty((frame_count - 2, point_count, 3, 3), dtype=np.float32)
    identity = np.eye(3, dtype=np.float64)

    for frame in range(frame_count - 2):
        particle_x = positions[frame]
        particle_v = (positions[frame + 2] - positions[frame]) / (2.0 * dt)
        particle_f = deformation[frame]
        particle_c = affine_velocity[frame]
        grid_pos = particle_x * inv_dx
        base = (grid_pos - 0.5).astype(np.int64)
        fx = grid_pos - base
        if (base < 0).any() or (base + 2 >= grid_size).any():
            raise ValueError(
                "grid transform places a particle outside the quadratic transfer domain"
            )
        weights = np.stack(
            (
                0.5 * np.square(1.5 - fx),
                0.75 - np.square(fx - 1.0),
                0.5 * np.square(fx - 0.5),
            ),
            axis=-1,
        )
        derivatives = np.stack((fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5), axis=-1)
        grid_mass = np.zeros(grid_size**3, dtype=np.float64)
        grid_velocity = np.zeros((grid_size**3, 3), dtype=np.float64)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = np.array([i, j, k])
                    flat_indices = (
                        (base[:, 0] + i) * grid_size * grid_size
                        + (base[:, 1] + j) * grid_size
                        + base[:, 2]
                        + k
                    )
                    weight = weights[:, 0, i] * weights[:, 1, j] * weights[:, 2, k]
                    dpos = (offset - fx) * dx
                    affine_term = np.einsum("nij,nj->ni", particle_c, dpos)
                    np.add.at(grid_mass, flat_indices, weight * volume)
                    np.add.at(
                        grid_velocity,
                        flat_indices,
                        weight[:, None] * volume[:, None] * (particle_v + affine_term),
                    )
        grid_velocity /= np.where(grid_mass > 1e-15, grid_mass, 1.0)[:, None]

        velocity_gradient = np.zeros((point_count, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    flat_indices = (
                        (base[:, 0] + i) * grid_size * grid_size
                        + (base[:, 1] + j) * grid_size
                        + base[:, 2]
                        + k
                    )
                    dweight = (
                        np.stack(
                            (
                                derivatives[:, 0, i]
                                * weights[:, 1, j]
                                * weights[:, 2, k],
                                weights[:, 0, i]
                                * derivatives[:, 1, j]
                                * weights[:, 2, k],
                                weights[:, 0, i]
                                * weights[:, 1, j]
                                * derivatives[:, 2, k],
                            ),
                            axis=1,
                        )
                        * inv_dx
                    )
                    velocity_gradient += (
                        grid_velocity[flat_indices, :, None] * dweight[:, None, :]
                    )
        predicted = (identity[None] + velocity_gradient * dt) @ particle_f
        residuals[frame] = np.abs(predicted - deformation[frame + 1]).astype(np.float32)
    return residuals


def _replace_dataset(output: h5py.File, name: str, value: np.ndarray) -> None:
    if name in output:
        del output[name]
    output.create_dataset(name, data=value, dtype=np.float32)


def _write_fields(
    path: Path,
    deformation: np.ndarray,
    affine_velocity: np.ndarray,
    baseline: np.ndarray,
    origin: np.ndarray,
    scale: float,
    *,
    neighbors: int,
    grid_size: int,
) -> None:
    point_count = deformation.shape[1]
    with h5py.File(path, "r+") as output:
        _replace_dataset(output, "deform_F", deformation[:, None])
        _replace_dataset(output, "deform_C", affine_velocity[:, None])
        _replace_dataset(
            output, "deform_volume", np.ones((1, point_count), dtype=np.float32)
        )
        _replace_dataset(output, "deform_baseline", baseline[:, None])
        _replace_dataset(output, "deform_grid_origin", origin.astype(np.float32))
        _replace_dataset(
            output, "deform_grid_scale", np.asarray([scale], dtype=np.float32)
        )
        output.attrs["deform_dt"] = DEFORM_DT
        output.attrs["deform_grid_size"] = grid_size
        output.attrs["deform_grid_lim"] = DEFORM_GRID_LIM
        output.attrs["deform_neighbors"] = neighbors


def derive_sample_fields(
    sample_dir: Path, neighbors: int = 32, *, grid_size: int = DEFORM_GRID_SIZE
) -> None:
    """Augment every object HDF5 in one TD sample with aligned deformation supervision."""
    sample_dir = Path(sample_dir)
    trajectories = _load_trajectories(sample_dir)
    frame_counts = {trajectory.shape[0] for _, trajectory in trajectories}
    if len(frame_counts) != 1:
        raise ValueError(f"{sample_dir}: objects must have the same frame count")
    origin, scale = make_shared_grid_transform(
        [trajectory for _, trajectory in trajectories], grid_size=grid_size
    )
    for path, trajectory in trajectories:
        grid_positions = world_to_grid(trajectory, origin, scale, grid_size=grid_size)
        deformation, affine_velocity = estimate_mls_fields(
            grid_positions, neighbors=neighbors
        )
        baseline = physctrl_update_residual(
            grid_positions,
            deformation,
            affine_velocity,
            np.ones(trajectory.shape[1], dtype=np.float64),
            grid_size=grid_size,
        )
        _write_fields(
            path,
            deformation,
            affine_velocity,
            baseline,
            origin,
            scale,
            neighbors=neighbors,
            grid_size=grid_size,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=DEFORM_GRID_SIZE)
    parser.add_argument(
        "--verify", action="store_true", help="print the stored GT raw residual summary"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    derive_sample_fields(
        args.sample_dir, neighbors=args.neighbors, grid_size=args.grid_size
    )
    if args.verify:
        for path in _object_paths(args.sample_dir):
            with h5py.File(path, "r") as source:
                raw = np.asarray(source["deform_baseline"][:])
                print(
                    f"{path.parent.name}: raw_mean={raw.mean():.6g} corrected_gt_mean=0"
                )


if __name__ == "__main__":
    main()
