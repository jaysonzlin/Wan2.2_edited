from training.overfit_config import load_config


def test_config_has_native_history_overfit_settings() -> None:
    config = load_config("configs/train/overfit_simgen_i2v_480_history.yaml", [])

    assert config["data"]["sample_root"].endswith("panda_ball_can")
    assert config["data"]["num_samples"] == 128
    assert (config["data"]["width"], config["data"]["height"]) == (480, 480)
    assert config["training"]["train_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["max_train_steps"] == 40_000
    assert config["training"]["num_train_timesteps"] == 10_000
    assert config["training"]["lr_scheduler"] == "constant"
    assert config["training"]["checkpoint_every_steps"] == 1_000
    assert config["training"]["visualization_every_steps"] == 500
    assert config["training"]["checkpoints_total_limit"] == 2
    assert (
        config["logging"]["output_dir"]
        == "outputs/simgen_i2v_480_history_first128_8gpu_40k"
    )
    assert config["validation"]["enabled"] is False
