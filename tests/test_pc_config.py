import pytest

from training.pc_config import load_pc_config


def test_pc_config_accepts_the_fixed_contract(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  dataset_root: training_dataset\n  num_frames: 49\n  num_points: 2048\n"
        "model:\n  n_layers: 8\n  latent_dim: 256\n  num_heads: 4\n"
        "flow:\n  prediction_type: flow\n  time_shift: 5.0\n"
        "lr_scheduler: cosine\n"
    )

    assert load_pc_config(path, [])["flow"]["prediction_type"] == "flow"


def test_pc_config_rejects_wrong_frame_count(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  num_frames: 48\n  num_points: 2048\n"
        "model:\n  n_layers: 8\n  latent_dim: 256\n  num_heads: 4\n"
        "flow:\n  prediction_type: flow\n  time_shift: 5\n"
    )

    with pytest.raises(ValueError, match="data.num_frames must be 49"):
        load_pc_config(path, [])


def test_pc_config_rejects_unknown_learning_rate_scheduler(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  num_frames: 49\n  num_points: 2048\n"
        "model:\n  n_layers: 8\n  latent_dim: 256\n  num_heads: 4\n"
        "flow:\n  prediction_type: flow\n  time_shift: 5\n"
        "lr_scheduler: triangular\n"
    )

    with pytest.raises(ValueError, match="lr_scheduler must be 'cosine' or 'constant'"):
        load_pc_config(path, [])
