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
