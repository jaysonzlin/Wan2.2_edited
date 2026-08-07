import json

import h5py
import numpy as np
import pytest
from PIL import Image

from training.joint_config import load_joint_config
from training.joint_dataset import JointWanPhysCtrlDataset, joint_collate


def _write_sample(root, name, object_names=("010", "002"), frames=49):
    sample = root / name
    sample.mkdir()
    for frame in range(frames):
        Image.new("RGBA", (2, 2), color=(frame, 0, 0, 255)).save(
            sample / f"rgba_{frame:05d}.png"
        )
    (sample / "metadata.json").write_text(json.dumps({"scene": name}))
    for object_name in object_names:
        object_dir = sample / "objects" / object_name
        object_dir.mkdir(parents=True)
        value = float(int(object_name))
        with h5py.File(object_dir / "pc.hdf5", "w") as source:
            source.create_dataset(
                "point_cloud",
                data=np.full((49, 1, 2, 3), value, dtype=np.float32),
            )
            source.create_dataset(
                "initial_linear_velocity", data=np.full((1, 3), value, dtype=np.float32)
            )
            source.create_dataset(
                "initial_angular_velocity",
                data=np.full((1, 3), -value, dtype=np.float32),
            )
    return sample


def _add_deformation_contract(path, points=2):
    with h5py.File(path, "r+") as source:
        source.create_dataset(
            "deform_F",
            data=np.broadcast_to(np.eye(3, dtype=np.float32), (49, 1, points, 3, 3)),
        )
        source.create_dataset(
            "deform_C", data=np.zeros((49, 1, points, 3, 3), dtype=np.float32)
        )
        source.create_dataset(
            "deform_volume", data=np.ones((1, points), dtype=np.float32)
        )
        source.create_dataset(
            "deform_baseline", data=np.zeros((47, 1, points, 3, 3), dtype=np.float32)
        )
        source.create_dataset(
            "deform_grid_origin", data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        source.create_dataset(
            "deform_grid_scale", data=np.array([0.5], dtype=np.float32)
        )
        source.attrs["deform_dt"] = 0.02
        source.attrs["deform_grid_size"] = 8
        source.attrs["deform_grid_lim"] = 10.0
        source.attrs["deform_neighbors"] = 1


def _valid_config() -> str:
    return """
data:
  dataset_root: training_dataset
  num_frames: 49
  num_points: 2048
  train_batch_size: 1
  max_objects_per_sample: 4
model:
  wan_cross_attention_blocks: 8
  physctrl_blocks: 8
  interaction_dim: 512
  interaction_heads: 8
objective:
  pc_type: ddpm_x0
  text_dropout_probability: 0
optimizer:
  video:
    lr: 1.0e-5
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1
  bca:
    lr: 1.0e-5
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1
  pc:
    lr: 1.0e-4
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 1.0e-2
training:
  denoised_latent_mse_every_steps: 50
  checkpoints_total_limit: 2
  resume_from_checkpoint: null
"""


def test_joint_dataset_loads_video_and_lexically_sorted_objects(tmp_path):
    _write_sample(tmp_path, "sample_0001")

    sample = JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)[
        0
    ]

    assert sample["sample_id"] == "sample_0001"
    assert sample["video"].shape == (49, 3, 2, 2)
    assert sample["object_ids"] == ["002", "010"]
    assert sample["point_clouds"].shape == (2, 49, 1, 2, 3)
    assert sample["point_clouds"][0].eq(2).all()
    assert sample["initial_linear_velocities"][1].eq(10).all()
    assert sample["initial_angular_velocities"][0].eq(-2).all()
    assert sample["metadata"] == {"scene": "sample_0001"}


def test_joint_dataset_allows_variable_numbers_of_objects(tmp_path):
    _write_sample(tmp_path, "sample_0001", object_names=("000",))
    _write_sample(tmp_path, "sample_0002", object_names=("000", "001", "002"))

    dataset = JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)

    assert dataset[0]["point_clouds"].shape[0] == 1
    assert dataset[1]["point_clouds"].shape[0] == 3


def test_joint_dataset_reads_enabled_deformation_contract(tmp_path):
    sample_dir = _write_sample(tmp_path, "sample_0001", object_names=("000",))
    _add_deformation_contract(sample_dir / "objects/000/pc.hdf5")

    sample = JointWanPhysCtrlDataset(
        tmp_path, expected_size=(2, 2), expected_points=2, load_deformation_fields=True
    )[0]

    assert sample["deform_F"].shape == (1, 49, 1, 2, 3, 3)
    assert sample["deform_C"].shape == (1, 49, 1, 2, 3, 3)
    assert sample["deform_volume"].shape == (1, 1, 2)
    assert sample["deform_baseline"].shape == (1, 47, 1, 2, 3, 3)
    assert sample["deform_grid_origin"].shape == (1, 3)
    assert sample["deform_grid_scale"].shape == (1, 1)


def test_joint_dataset_rejects_missing_enabled_deformation_contract(tmp_path):
    _write_sample(tmp_path, "sample_0001", object_names=("000",))

    with pytest.raises(KeyError, match="deform_F"):
        JointWanPhysCtrlDataset(
            tmp_path,
            expected_size=(2, 2),
            expected_points=2,
            load_deformation_fields=True,
        )


def test_joint_collate_keeps_unrestricted_metadata_and_adds_only_the_video_batch_axis(
    tmp_path,
):
    _write_sample(tmp_path, "sample_0001", object_names=("000",))
    sample = JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)[
        0
    ]
    sample["metadata"] = {"different_length_lists": [[1], [2, 3]]}

    batch = joint_collate([sample])

    assert batch["point_clouds"].shape == (1, 1, 49, 1, 2, 3)
    assert batch["metadata"] == sample["metadata"]


def test_joint_dataset_rejects_bad_metadata_and_bad_trajectory_frames(tmp_path):
    sample = _write_sample(tmp_path, "sample_0001")
    (sample / "metadata.json").write_text("[]")
    with pytest.raises(ValueError, match="metadata.json must contain a JSON object"):
        JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)

    (sample / "metadata.json").write_text("{}")
    with h5py.File(sample / "objects" / "002" / "pc.hdf5", "r+") as source:
        del source["point_cloud"]
        source.create_dataset(
            "point_cloud", data=np.zeros((48, 1, 2, 3), dtype=np.float32)
        )
    with pytest.raises(
        ValueError, match=r"point_cloud must have shape \(49, 1, 2, 3\)"
    ):
        JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)


def test_joint_dataset_rejects_missing_video_frame(tmp_path):
    sample = _write_sample(tmp_path, "sample_0001")
    (sample / "rgba_00048.png").unlink()

    with pytest.raises(ValueError, match="missing required frame rgba_00048.png"):
        JointWanPhysCtrlDataset(tmp_path, expected_size=(2, 2), expected_points=2)


def test_joint_config_accepts_joint_contract_and_rejects_batching(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(_valid_config())

    assert load_joint_config(path, [])["model"]["interaction_dim"] == 512

    path.write_text(
        _valid_config().replace("train_batch_size: 1", "train_batch_size: 2")
    )
    with pytest.raises(ValueError, match="data.train_batch_size must be 1"):
        load_joint_config(path, [])


def test_joint_config_accepts_separate_optimizer_groups(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(_valid_config())

    optimizer = load_joint_config(path, ["optimizer.pc.lr=2.0e-4"])["optimizer"]

    assert optimizer["video"]["lr"] == 1.0e-5
    assert optimizer["bca"]["betas"] == [0.9, 0.95]
    assert optimizer["pc"]["lr"] == 2.0e-4


def test_joint_config_accepts_rigid_loss_settings(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            "  text_dropout_probability: 0\n  rigid_loss_weight: 0.25\n  rigid_loss_neighbors: 8",
        )
    )

    objective = load_joint_config(path, [])["objective"]

    assert objective["rigid_loss_weight"] == 0.25
    assert objective["rigid_loss_neighbors"] == 8


def test_joint_config_accepts_boolean_rigid_loss_toggle(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            "  text_dropout_probability: 0\n  enable_rigid_loss: true",
        )
    )

    assert load_joint_config(path, [])["objective"]["enable_rigid_loss"] is True


def test_joint_config_accepts_deform_loss_settings(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            "  text_dropout_probability: 0\n  enable_deform_loss: true\n  deform_loss_weight: 0.001\n  deform_loss_neighbors: 32",
        )
    )

    objective = load_joint_config(path, [])["objective"]

    assert objective["enable_deform_loss"] is True
    assert objective["deform_loss_weight"] == 0.001
    assert objective["deform_loss_neighbors"] == 32


@pytest.mark.parametrize("value", ["1", '"true"', "null"])
def test_joint_config_rejects_non_boolean_rigid_loss_toggle(tmp_path, value):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            f"  text_dropout_probability: 0\n  enable_rigid_loss: {value}",
        )
    )

    with pytest.raises(ValueError, match="enable_rigid_loss"):
        load_joint_config(path, [])


@pytest.mark.parametrize("value", ["1", '"true"', "null"])
def test_joint_config_rejects_non_boolean_deform_loss_toggle(tmp_path, value):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            f"  text_dropout_probability: 0\n  enable_deform_loss: {value}",
        )
    )

    with pytest.raises(ValueError, match="enable_deform_loss"):
        load_joint_config(path, [])


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("rigid_loss_weight", "-0.1", "rigid_loss_weight"),
        ("rigid_loss_weight", "true", "rigid_loss_weight"),
        ("rigid_loss_neighbors", "0", "rigid_loss_neighbors"),
        ("rigid_loss_neighbors", "2048", "rigid_loss_neighbors"),
        ("rigid_loss_neighbors", "true", "rigid_loss_neighbors"),
    ],
)
def test_joint_config_rejects_invalid_rigid_loss_settings(
    tmp_path, setting, value, message
):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "  text_dropout_probability: 0",
            f"  text_dropout_probability: 0\n  {setting}: {value}",
        )
    )

    with pytest.raises(ValueError, match=message):
        load_joint_config(path, [])


def test_joint_config_rejects_invalid_resume_and_checkpoint_metric_settings(tmp_path):
    path = tmp_path / "joint.yaml"
    path.write_text(
        _valid_config().replace(
            "denoised_latent_mse_every_steps: 50", "denoised_latent_mse_every_steps: 0"
        )
    )
    with pytest.raises(
        ValueError, match="denoised_latent_mse_every_steps must be a positive integer"
    ):
        load_joint_config(path, [])

    path.write_text(
        _valid_config().replace(
            "checkpoints_total_limit: 2", "checkpoints_total_limit: 0"
        )
    )
    with pytest.raises(
        ValueError, match="checkpoints_total_limit must be a positive integer"
    ):
        load_joint_config(path, [])

    path.write_text(
        _valid_config().replace(
            "resume_from_checkpoint: null", "resume_from_checkpoint: 123"
        )
    )
    with pytest.raises(
        ValueError,
        match="resume_from_checkpoint must be null, 'latest', or a path string",
    ):
        load_joint_config(path, [])
