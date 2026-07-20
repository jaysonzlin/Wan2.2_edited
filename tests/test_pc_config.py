import pytest

from training.pc_config import load_pc_config


def valid_config_text(**model_overrides) -> str:
    model = {
        "n_layers": 8,
        "latent_dim": 256,
        "num_heads": 4,
        "point_embed": True,
        "frame_cond": True,
        "transformer_block": "SpatialTemporalTransformerBlock",
    }
    model.update(model_overrides)
    model_lines = "".join(
        f"  {key}: {str(value).lower() if isinstance(value, bool) else value}\n"
        for key, value in model.items()
    )
    return (
        "data:\n  dataset_root: training_dataset\n  num_frames: 49\n  num_points: 2048\n"
        f"model:\n{model_lines}"
        "objective:\n  type: ddpm\n  num_train_timesteps: 1000\n"
        "  beta_schedule: linear\n  time_shift: 5.0\n"
        "lr_scheduler: cosine\n"
    )


def test_pc_config_accepts_the_fixed_contract(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text())

    assert load_pc_config(path, [])['objective']['type'] == "ddpm"


def test_pc_config_rejects_wrong_frame_count(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text().replace("num_frames: 49", "num_frames: 48"))

    with pytest.raises(ValueError, match="data.num_frames must be 49"):
        load_pc_config(path, [])


def test_pc_config_rejects_unknown_learning_rate_scheduler(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text().replace("lr_scheduler: cosine", "lr_scheduler: triangular"))

    with pytest.raises(ValueError, match="lr_scheduler must be 'cosine' or 'constant'"):
        load_pc_config(path, [])


@pytest.mark.parametrize(
    ("model_overrides", "message"),
    [
        ({"point_embed": False}, "model.point_embed must be true"),
        ({"frame_cond": False}, "model.frame_cond must be true"),
        (
            {"transformer_block": "TemporalOnlyTransformerBlock"},
            "model.transformer_block must be 'SpatialTemporalTransformerBlock'",
        ),
        ({"num_heads": 2}, "model.num_heads must equal model.latent_dim // 64"),
    ],
)
def test_pc_config_rejects_non_physctrl_model_option(tmp_path, model_overrides, message):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text(**model_overrides))

    with pytest.raises(ValueError, match=message):
        load_pc_config(path, [])
