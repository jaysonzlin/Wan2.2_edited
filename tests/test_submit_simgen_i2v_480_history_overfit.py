import subprocess
from pathlib import Path


def test_submit_script_is_valid_one_h200_requeue_launcher() -> None:
    script_path = Path("submit_simgen_i2v_480_history_overfit.sh")

    result = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
    script = script_path.read_text()

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=gpu_requeue" in script
    assert "#SBATCH --constraint=h200" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --requeue" in script
    assert "configs/accelerate/h200_single_gpu.yaml" in script
    assert "train_i2v_simgen_480_overfit.py" in script
    assert "training.resume_from_checkpoint=latest" in script
