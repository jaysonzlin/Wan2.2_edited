from pathlib import Path


def test_frozen_auxiliary_training_launcher_uses_the_isolated_4000_step_config():
    script = Path(
        "submit_joint_simgen_8gpu_2node_mamba_video_only_3slot_frozen_auxiliary_h200_4k.sh"
    ).read_text()

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --gres=gpu:nvidia_h200:4" in script
    assert "joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k" in script
    assert "configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml" in script
    assert "joint_simgen_video_only.py" in script
    assert "training.resume_from_checkpoint" not in script
    assert "training.max_train_steps=" not in script
    assert "Frozen auxiliary modules; resume latest; 4000 optimizer steps" in script


def test_frozen_auxiliary_validation_launcher_targets_checkpoint_4000():
    script = Path(
        "submit_visualize_joint_simgen_video_only_frozen_auxiliary_validation.sh"
    ).read_text()

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:nvidia_h200:1" in script
    assert "joint_simgen_video_only_3slot_frozen_auxiliary_h200_4k/checkpoint-4000" in script
    assert "visualize_joint_simgen_video_only_validation.py" in script
    assert "--overwrite" in script


def test_one_gpu_frozen_auxiliary_diagnostic_launcher_runs_one_real_optimizer_step():
    script = Path(
        "submit_joint_simgen_video_only_1gpu_frozen_auxiliary_diagnostic.sh"
    ).read_text()

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:nvidia_h200:1" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "joint_simgen_video_only_1gpu_diagnostic.py" in script
    assert (
        "--config "
        "configs/train/joint_simgen_480_8gpu_video_only_3slot_frozen_auxiliary.yaml"
        in script
    )
    assert "--frozen-auxiliary" in script


def test_two_gpu_diagnostic_uses_accelerate_for_two_frozen_auxiliary_steps():
    script = Path("submit_joint_simgen_video_only_2gpu_diagnostic.sh").read_text()

    assert "torchrun" not in script
    assert "accelerate" in script
    assert "configs/accelerate/h200_2gpu.yaml" in script
    assert "--expected-processes 2" in script
    assert "--optimizer-steps 2" in script
    assert "--frozen-auxiliary" in script
