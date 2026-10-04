"""Render fixed SimGen validation predictions and targets from a joint checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_config(path: Path, overrides: list[str]) -> dict:
    """Load the recorded joint-training configuration only when rendering starts."""
    from training.simgen_joint_config import load_simgen_joint_config

    return load_simgen_joint_config(path, overrides)


def render_validation(config: dict, checkpoint: Path, output_dir: Path) -> None:
    """Load the CUDA-dependent renderer only after CLI validation succeeds."""
    from joint_simgen import visualize_validation

    visualize_validation(config, checkpoint, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        help="Configuration to use; defaults to config.yaml beside the checkpoint.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rendering into a populated output directory.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _default_output_dir(checkpoint: Path) -> Path:
    return checkpoint.parent / f"{checkpoint.name}-validation"


def _ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and (output_dir.is_file() or any(output_dir.iterdir())):
        if not overwrite:
            raise FileExistsError(
                f"validation output already exists: {output_dir}; pass --overwrite to reuse it"
            )


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "config.yaml"
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(checkpoint)
    _ensure_output_dir(output_dir, overwrite=args.overwrite)
    config = load_config(config_path, args.overrides)
    render_validation(config, checkpoint, output_dir)


if __name__ == "__main__":
    main()
