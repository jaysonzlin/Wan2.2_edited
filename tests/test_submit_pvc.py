from pathlib import Path


def test_pvc_launcher_uses_the_separate_gate_centroid_mse_experiment():
    script = Path("submit_pvc.sh").read_text()

    assert "--config configs/train/config_pvc_utonia_history_overfit.yaml" in script
    assert "output_dir=./outputs/pvc_trajectory_utonia_history_centroid_mse" in script
    assert "tracker_project_name=pvc_trajectory_utonia_history_centroid_mse" in script
    assert "model.point_view_gate_mode=separate" in script
