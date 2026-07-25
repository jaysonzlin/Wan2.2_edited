from pathlib import Path


def test_launcher_uses_isolated_832x480_training_paths_and_lingbot_overrides():
    script = Path("submit_832x480.sh").read_text()

    assert "#SBATCH --job-name=wan_i2v_832x480" in script
    assert "logs/wan_i2v_832x480_%j.out" in script
    assert "train_i2v_832x480.py" in script
    assert "--config configs/train/overfit_kubric_i2v_832x480.yaml" in script
    assert "logging.output_dir=outputs/i2v_832x480" in script
    for override in (
        "training.max_train_steps=10000",
        "training.learning_rate=1.0e-5",
        "training.lr_scheduler=constant",
        "training.max_grad_norm=2.0",
        "training.weight_decay=0.1",
        "training.adam_beta1=0.9",
        "training.adam_beta2=0.95",
    ):
        assert override in script
