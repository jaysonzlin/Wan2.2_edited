from pathlib import Path


def test_submit_joint_requests_h200_and_launches_joint_trainer() -> None:
    script = Path("submit_joint.sh").read_text()

    for expected in (
        "#SBATCH --partition=gpu_h200",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=08:00:00",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "training.resume_from_checkpoint=latest",
    ):
        assert expected in script
