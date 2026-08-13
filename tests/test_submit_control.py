from pathlib import Path


def test_control_launcher_disables_utonia_and_isolates_its_resume_state():
    script = Path("submit_control.sh").read_text()

    for declaration in (
        "#SBATCH --job-name=control",
        "logs/control_%j.out",
        "logs/control_%j.err",
        "model.utonia_enabled=false",
        "output_dir=./outputs/pc_trajectory_control_overfit",
        "tracker_project_name=pc_trajectory_control_overfit",
        "resume_from_checkpoint=latest",
    ):
        assert declaration in script

    assert "--config configs/train/config_pc_utonia_overfit.yaml" in script
