"""Render fixed SimGen validation predictions and targets from a frozen-video checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from joint_simgen import visualize_frozen_video_validation
from training.simgen_joint_config import load_simgen_joint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint.parent / f"{checkpoint.name}-validation"
    )
    config = load_simgen_joint_config(args.config, args.overrides)
    visualize_frozen_video_validation(config, checkpoint, output_dir)


if __name__ == "__main__":
    main()
