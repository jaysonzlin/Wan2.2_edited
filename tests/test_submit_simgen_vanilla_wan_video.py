from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_vanilla_wan_launcher_uses_the_isolated_native_video_contract():
    script = PROJECT_DIR / "submit_simgen_8gpu_2node_mamba_vanilla_wan_video.sh"

    source = script.read_text()

    assert "#SBATCH --nodes=2" in source
    assert "#SBATCH --gres=gpu:nvidia_h200:4" in source
    assert "configs/accelerate/h200_8gpu_2node.yaml" in source
    assert "configs/train/simgen_480_8gpu_vanilla_wan_video.yaml" in source
    assert "simgen_vanilla_video.py" in source
    assert "load_simgen_vanilla_video_config" in source
    assert "training.resume_from_checkpoint" not in source
    assert "joint_simgen_video_only.py" not in source
