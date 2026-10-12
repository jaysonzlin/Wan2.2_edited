from pathlib import Path

from PIL import Image

from training.simgen_video_only_dataset import SimGenVideoDataset, simgen_video_collate


def _write_video_only_sample(root: Path, sample_id: str = "sample_0") -> None:
    view_root = root / sample_id / "view_0"
    view_root.mkdir(parents=True)
    for frame in range(49):
        Image.new("RGB", (480, 480), color=(frame, 2, 3)).save(
            view_root / f"{frame:08d}.png"
        )


def test_video_only_dataset_loads_frames_without_object_side_inputs(tmp_path):
    _write_video_only_sample(tmp_path)

    item = SimGenVideoDataset(tmp_path, sample_ids=[0])[0]

    assert item["sample_id"] == "sample_0"
    assert item["video"].shape == (49, 3, 480, 480)
    assert item["video"][0, :, 0, 0].tolist() == [
        -1.0,
        -0.9843137264251709,
        -0.9764705896377563,
    ]
    assert set(item) == {"sample_id", "video"}


def test_video_only_collate_adds_only_the_batch_dimension(tmp_path):
    _write_video_only_sample(tmp_path)
    item = SimGenVideoDataset(tmp_path, sample_ids=[0])[0]

    batch = simgen_video_collate([item])

    assert batch["sample_id"] == "sample_0"
    assert batch["video"].shape == (1, 49, 3, 480, 480)
