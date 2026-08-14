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

    assert '"${PROJECT_DIR}/cur.sif"' in script
    assert "--config_file configs/accelerate/h200_single_gpu.yaml" in script
    assert "train_pc.py" in script
    assert "--config configs/train/config_pc_utonia_overfit.yaml" in script
    assert "num_train_epochs=10000" in script
    assert "max_train_steps=10000" not in script
    assert "checkpoints_total_limit=2" in script
    assert "resume_from_checkpoint=latest" in script
    assert "training.resume_from_checkpoint=latest" not in script


def test_history_utonia_launcher_targets_history_experiment():
    script = Path("submit_history_utonia.sh").read_text()

    assert "#SBATCH --job-name=history_utonia" in script
    assert "logs/history_utonia_%j.out" in script
    assert "logs/history_utonia_%j.err" in script
    assert "--config configs/train/config_pc_utonia_history_overfit.yaml" in script
    assert "resume_from_checkpoint=latest" in script
