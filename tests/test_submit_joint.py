from pathlib import Path


def test_submit_joint_requests_h200_and_launches_joint_trainer() -> None:
    script = Path("submit_joint.sh").read_text()

    for expected in (
        "#SBATCH --partition=gpu_h200",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=10:30:00",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "training.resume_from_checkpoint=latest",
        "training.max_train_steps=10000",
    ):
        assert expected in script


def test_three_object_joint_launcher_uses_isolated_dataset_and_output() -> None:
    script = Path("submit_joint_3.sh").read_text()

    for expected in (
        "#SBATCH --job-name=wan_joint_physctrl_3",
        "logs/wan_joint_physctrl_3_%j.out",
        "logs/wan_joint_physctrl_3_%j.err",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "data.dataset_root=td_832x480_3",
        "logging.output_dir=outputs/joint_wan_physctrl_3",
        "training.resume_from_checkpoint=latest",
        "training.max_train_steps=10000",
    ):
        assert expected in script


def test_soft_three_object_joint_launcher_uses_soft_dataset_and_isolated_output() -> None:
    script = Path("submit_joint_soft_3.sh").read_text()

    for expected in (
        "#SBATCH --job-name=joint_soft_3",
        "#SBATCH --cpus-per-task=4",
        "logs/joint_soft_3_%j.out",
        "logs/joint_soft_3_%j.err",
        "--config_file configs/accelerate/h200_single_gpu.yaml",
        "train_joint_wan_physctrl.py",
        "--config configs/train/joint_wan_physctrl_832x480.yaml",
        "data.dataset_root=td_832x480_3_soft",
        "logging.output_dir=outputs/joint_soft_3",
        "training.resume_from_checkpoint=latest",
        "training.max_train_steps=10000",
    ):
        assert expected in script
