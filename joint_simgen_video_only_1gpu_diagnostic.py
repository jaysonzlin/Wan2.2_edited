"""Run one real video-only SimGen backward pass without distributed training."""

from __future__ import annotations

import argparse

from joint_simgen import _shared_generator
from joint_simgen_video_only import (
    _create_video_only_training_accelerator,
    _video_only_losses,
    build_datasets,
)
from training.simgen_video_only_config import load_simgen_video_only_config
from training.simgen_video_only_dataset import simgen_video_collate
from training.wan_i2v_training import load_frozen_encoders, load_trainable_dit


def main() -> None:
    """Exercise the production model and first real sample through one backward pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml",
    )
    parser.add_argument("--expected-processes", type=int, default=1)
    args = parser.parse_args()

    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
    from wan.modules.pc_trajectory import PCTrajectoryModel

    config = load_simgen_video_only_config(args.config, [])
    training, logging = config["training"], config["logging"]
    accelerator = _create_video_only_training_accelerator(
        training, {**logging, "report_to": None}
    )
    if accelerator.num_processes != args.expected_processes:
        raise RuntimeError(
            f"Expected {args.expected_processes} processes, found {accelerator.num_processes}"
        )

    set_seed(training["seed"], device_specific=False)
    train_dataset, _ = build_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=simgen_video_collate,
    )

    vae, text_encoder = load_frozen_encoders(
        config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device
    )
    model = JointWanPhysCtrlModel(
        load_trainable_dit(
            config["model"]["checkpoint_dir"],
            config["model"]["gradient_checkpointing"],
        ),
        PCTrajectoryModel(
            n_points=config["data"]["num_points"],
            n_future_frames=45,
            latent_dim=256,
            n_layers=8,
            num_heads=4,
            objective_type="ddpm",
            conditioning="history",
            history_frames=config["model"]["history_frames"],
            utonia_feature_dim=config["model"]["utonia_feature_dim"],
        ),
    )
    model, train_loader = accelerator.prepare(model, train_loader)
    model.train()
    generator = _shared_generator(accelerator.device, training["seed"])
    batch = next(iter(train_loader))
    print(
        "DIAGNOSTIC: "
        f"rank={accelerator.process_index} loaded production batch "
        f"video_shape={tuple(batch['video'].shape)} device={accelerator.device}",
        flush=True,
    )

    print(
        f"DIAGNOSTIC: rank={accelerator.process_index} starting production-shaped forward",
        flush=True,
    )
    loss, _, _ = _video_only_losses(
        batch,
        model,
        vae,
        text_encoder,
        generator,
        accelerator.device,
        config["data"],
        config["model"],
        config["objective"],
    )
    print(
        f"DIAGNOSTIC: rank={accelerator.process_index} "
        f"forward completed loss={loss.detach().item():.6f}",
        flush=True,
    )
    print(
        f"DIAGNOSTIC: rank={accelerator.process_index} starting video-only backward",
        flush=True,
    )
    accelerator.backward(loss)

    unwrapped = accelerator.unwrap_model(model)
    if not any(parameter.grad is not None for parameter in unwrapped.wan_model.parameters()):
        raise AssertionError("video-only backward did not reach Wan parameters")
    print(
        f"PASS: rank={accelerator.process_index} production-shaped video-only backward completed",
        flush=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
