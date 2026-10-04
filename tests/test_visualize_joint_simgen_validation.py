import importlib
import sys
from pathlib import Path

import pytest


def _load_module():
    sys.modules.pop("visualize_joint_simgen_validation", None)
    return importlib.import_module("visualize_joint_simgen_validation")


def test_main_uses_checkpoint_parent_config_and_output_defaults(monkeypatch, tmp_path):
    """A checkpoint renders with the exact configuration recorded by its run."""
    checkpoint = tmp_path / "checkpoint-9000"
    checkpoint.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n")
    renderer = _load_module()
    loaded = []
    rendered = []
    monkeypatch.setattr(
        renderer,
        "load_config",
        lambda path, overrides: loaded.append((Path(path), overrides)) or {"loaded": True},
    )
    monkeypatch.setattr(
        renderer,
        "render_validation",
        lambda config, path, output: rendered.append((config, Path(path), Path(output))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_joint_simgen_validation.py", "--checkpoint", str(checkpoint)],
    )

    renderer.main()

    assert loaded == [(config_path, [])]
    assert rendered == [
        ({"loaded": True}, checkpoint, tmp_path / "checkpoint-9000-validation")
    ]


def test_main_rejects_populated_output_without_overwrite(monkeypatch, tmp_path):
    """A rerun cannot silently mix old and newly rendered validation artifacts."""
    checkpoint = tmp_path / "checkpoint-9000"
    checkpoint.mkdir()
    (tmp_path / "config.yaml").write_text("data: {}\n")
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    (output_dir / "old-video.mp4").write_text("old output\n")
    renderer = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_joint_simgen_validation.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        renderer.main()
