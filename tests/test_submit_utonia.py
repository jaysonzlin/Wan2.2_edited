from pathlib import Path


def test_utonia_launcher_uses_a100_requeue_and_latest_resume():
    script = Path("submit_utonia.sh").read_text()

    for declaration in (
        "#SBATCH --job-name=utonia",
        "#SBATCH --partition=gpu_requeue",
        "#SBATCH --constraint=a100",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=64G",
        "#SBATCH --time=10:30:00",
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
        "logs/utonia_%j.out",
        "logs/utonia_%j.err",
    ):
        assert declaration in script

    assert '"${PROJECT_DIR}/current.sif"' in script
    assert "--config_file configs/accelerate/h200_single_gpu.yaml" in script
    assert "train_pc.py" in script
    assert "--config configs/train/config_pc_utonia_overfit.yaml" in script
    assert "training.resume_from_checkpoint=latest" in script
