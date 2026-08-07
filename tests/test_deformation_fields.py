import h5py
import numpy as np
import pytest

from training.derive_deformation_fields import derive_sample_fields


def _write_td_sample(root, object_count=2, frames=49, points=8):
    """Create a tiny, non-coplanar TD-style sample with elastic motion."""
    rest = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )[:points]
    for object_index in range(object_count):
        object_dir = root / "objects" / f"{object_index:03d}"
        object_dir.mkdir(parents=True)
        trajectory = []
        for frame in range(frames):
            deformation = np.diag(
                [1.0 + 0.001 * frame, 1.0 - 0.0005 * frame, 1.0 + 0.00075 * frame]
            ).astype(np.float32)
            trajectory.append(
                rest @ deformation.T + np.array([object_index, 0.01 * frame, 0.0])
            )
        with h5py.File(object_dir / "pc.hdf5", "w") as output:
            output.create_dataset("point_cloud", data=np.asarray(trajectory)[:, None])
            output.create_dataset(
                "initial_linear_velocity", data=np.zeros((1, 3), dtype=np.float32)
            )
            output.create_dataset(
                "initial_angular_velocity", data=np.zeros((1, 3), dtype=np.float32)
            )


def test_derive_sample_fields_writes_aligned_deformation_arrays(tmp_path):
    _write_td_sample(tmp_path)

    derive_sample_fields(tmp_path, neighbors=3, grid_size=8)

    for object_name in ("000", "001"):
        with h5py.File(tmp_path / "objects" / object_name / "pc.hdf5") as source:
            assert source["deform_F"].shape == (49, 1, 8, 3, 3)
            assert source["deform_C"].shape == (49, 1, 8, 3, 3)
            assert source["deform_volume"].shape == (1, 8)
            assert source["deform_baseline"].shape == (47, 1, 8, 3, 3)
            assert source["deform_grid_origin"].shape == (3,)
            assert source["deform_grid_scale"].shape == (1,)
            assert source.attrs["deform_dt"] == pytest.approx(0.02)
            assert source.attrs["deform_grid_size"] == 8
            assert source.attrs["deform_grid_lim"] == pytest.approx(10.0)
            assert source.attrs["deform_neighbors"] == 3
            assert np.isfinite(source["deform_F"][:]).all()
            assert np.isfinite(source["deform_C"][:]).all()
            assert np.isfinite(source["deform_baseline"][:]).all()


def test_derive_sample_fields_uses_one_shared_grid_transform(tmp_path):
    _write_td_sample(tmp_path)

    derive_sample_fields(tmp_path, neighbors=3, grid_size=8)

    with (
        h5py.File(tmp_path / "objects/000/pc.hdf5") as first,
        h5py.File(tmp_path / "objects/001/pc.hdf5") as second,
    ):
        assert np.array_equal(
            first["deform_grid_origin"][:], second["deform_grid_origin"][:]
        )
        assert np.array_equal(
            first["deform_grid_scale"][:], second["deform_grid_scale"][:]
        )
