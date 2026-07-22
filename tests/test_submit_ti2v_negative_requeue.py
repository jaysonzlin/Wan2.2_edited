from pathlib import Path


def test_ti2v_negative_requeue_script_uses_requested_prompt_and_output() -> None:
    script = Path("submit_h200_ti2v_negative_requeue.sh").read_text()

    assert "#SBATCH --job-name=wan_ti2v" in script
    assert "train_i2v.py" in script
    assert 'data.prompt="Colorful object falls onto the light gray ground"' in script
    assert "training.use_wan_negative_prompt=true" in script
    assert "logging.output_dir=outputs/ti2v_negative_requeue" in script
    assert "train_pc.py" not in script
