import numpy as np

from training.pc_visualization import save_pointcloud_comparison_mp4


def test_comparison_visualization_writes_mp4(tmp_path):
    trajectory = np.zeros((2, 1, 2, 3), dtype=np.float32)
    output = tmp_path / "comparison.mp4"

    save_pointcloud_comparison_mp4(trajectory, trajectory, output, fps=1)

    assert output.is_file()
    assert output.stat().st_size > 0
