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
        "lr_scheduler: cosine\ncheckpoints_total_limit: 2\n"
    )


def test_pc_config_accepts_the_fixed_contract(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text())

    assert load_pc_config(path, [])['objective']['type'] == "ddpm"


def test_pc_config_accepts_baseline_without_utonia_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text())

    config = load_pc_config(path, [])

    assert config["model"].get("utonia_enabled") is None


@pytest.mark.parametrize("conditioning_addition", ["", "  conditioning: velocity\n"])
def test_pc_config_accepts_default_and_explicit_velocity_conditioning(
    tmp_path, conditioning_addition
):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "objective:\n", conditioning_addition + "objective:\n"
        )
    )

    assert load_pc_config(path, [])["objective"]["type"] == "ddpm"


def test_pc_config_accepts_four_frame_history_conditioning(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "objective:\n", "  conditioning: history\n  history_frames: 4\nobjective:\n"
        )
    )

    assert load_pc_config(path, [])["model"]["history_frames"] == 4


@pytest.mark.parametrize(
    ("conditioning_addition", "message"),
    [
        (
            "  conditioning: acceleration\n",
            "model.conditioning must be 'velocity' or 'history'",
        ),
        (
            "  conditioning: history\n",
            "model.history_frames must be 4 when conditioning is 'history'",
        ),
        (
            "  conditioning: history\n  history_frames: 3\n",
            "model.history_frames must be 4 when conditioning is 'history'",
        ),
        (
            "  conditioning: history\n  history_frames: 4.0\n",
            "model.history_frames must be 4 when conditioning is 'history'",
        ),
        (
            "  conditioning: history\n  history_frames: four\n",
            "model.history_frames must be 4 when conditioning is 'history'",
        ),
    ],
)
def test_pc_config_rejects_invalid_history_conditioning_fields(
    tmp_path, conditioning_addition, message
):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "objective:\n", conditioning_addition + "objective:\n"
        )
    )

    with pytest.raises(ValueError, match=message):
        load_pc_config(path, [])


@pytest.mark.parametrize(
    ("data_addition", "message"),
    [
        ("", "data.object_id"),
        (
            "  object_id: '000'\n",
            "data.utonia_cache_root",
        ),
    ],
)
def test_utonia_config_requires_object_id_and_cache_root(tmp_path, data_addition, message):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "model:\n", f"{data_addition}model:\n  utonia_enabled: true\n"
        )
    )

    with pytest.raises(ValueError, match=message):
        load_pc_config(path, [])


def test_utonia_config_rejects_non_boolean_enabled_flag(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text().replace("objective:\n", "  utonia_enabled: enabled\nobjective:\n"))

    with pytest.raises(ValueError, match="model.utonia_enabled must be a boolean"):
        load_pc_config(path, [])


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


@pytest.mark.parametrize("resume", ["latest", "outputs/run/checkpoint-250"])
def test_pc_config_accepts_resume_setting(tmp_path, resume):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text() + f"resume_from_checkpoint: {resume}\n")

    assert load_pc_config(path, [])["resume_from_checkpoint"] == resume


@pytest.mark.parametrize("resume", ["0", "[]", "'   '"])
def test_pc_config_rejects_invalid_resume_setting(tmp_path, resume):
    path = tmp_path / "config.yaml"
    path.write_text(valid_config_text() + f"resume_from_checkpoint: {resume}\n")

    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        load_pc_config(path, [])


@pytest.mark.parametrize("limit", ["0", "-1", "1.5", "null", "invalid"])
def test_pc_config_rejects_invalid_checkpoint_retention_limit(tmp_path, limit):
    path = tmp_path / "config.yaml"
    path.write_text(
        valid_config_text().replace(
            "checkpoints_total_limit: 2\n",
            f"checkpoints_total_limit: {limit}\n",
        )
    )

    with pytest.raises(ValueError, match="checkpoints_total_limit must be a positive integer"):
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
