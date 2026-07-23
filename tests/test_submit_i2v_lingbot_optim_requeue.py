from pathlib import Path


def test_i2v_lingbot_optimizer_requeue_script_uses_requested_overrides() -> None:
    script = Path("submit_h200_i2v_lingbot_optim_requeue.sh").read_text()

    assert "#SBATCH --job-name=wan_i2v_lingbot_optim" in script
    assert "#SBATCH --partition=gpu_h200" in script
    assert "#SBATCH --constraint=h200" not in script
    assert "#SBATCH --requeue" not in script
    assert "train_i2v.py" in script
    assert 'data.prompt=""' in script
    assert "training.max_grad_norm=2.0" in script
    assert "training.weight_decay=0.1" in script
    assert "training.adam_beta1=0.9" in script
    assert "training.adam_beta2=0.95" in script
    assert "logging.output_dir=outputs/i2v_lingbot_optim_requeue" in script
    assert "train_pc.py" not in script
