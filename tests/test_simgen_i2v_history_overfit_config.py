from training.overfit_config import load_config


def test_config_has_native_history_overfit_settings() -> None:
    config = load_config("configs/train/overfit_simgen_i2v_480_history.yaml", [])

    assert config["data"]["sample_root"].endswith("sample_0/view_0")
    assert (config["data"]["width"], config["data"]["height"]) == (480, 480)
    assert config["training"]["max_train_steps"] == 10_000
    assert config["training"]["checkpoint_every_steps"] == 500
    assert config["training"]["visualization_every_steps"] == 500
    assert config["training"]["checkpoints_total_limit"] == 2
    assert config["validation"]["enabled"] is False
