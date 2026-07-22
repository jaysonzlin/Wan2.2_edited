from pathlib import Path


def test_i2v_requeue_script_uses_empty_prompt_and_isolated_output() -> None:
    script = Path("submit_h200_i2v_requeue.sh").read_text()

    assert "#SBATCH --job-name=wan_i2v" in script
    assert "train_i2v.py" in script
    assert 'data.prompt=""' in script
    assert "logging.output_dir=outputs/i2v_requeue" in script
    assert "train_pc.py" not in script
