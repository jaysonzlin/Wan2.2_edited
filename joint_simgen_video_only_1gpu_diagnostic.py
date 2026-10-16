"""Run one real video-only SimGen backward pass without distributed training."""

from __future__ import annotations

import argparse

import torch

from joint_simgen import _shared_generator
from joint_simgen_video_only import (
    _create_video_only_training_accelerator,
    _video_only_losses,
    assert_frozen_pc_and_bridges_have_no_gradients,
    build_datasets,
    create_frozen_auxiliary_optimizer,
    freeze_pc_and_bridges,
)
from training.simgen_video_only_config import load_simgen_video_only_config
from training.simgen_video_only_dataset import simgen_video_collate
from training.wan_i2v_training import load_frozen_encoders, load_trainable_dit
from train_joint_wan_physctrl import (
    _bridge_gradient_norm,
    pc_gradient_norm,
    video_gradient_norm,
)


def capture_wan_update_probe(model):
    """Capture one Wan parameter element with a nonzero gradient before an update."""
    for parameter in model.wan_model.parameters():
        if parameter.grad is None:
            continue
        flattened_gradient = parameter.grad.detach().flatten()
        flat_index = int(flattened_gradient.abs().argmax().item())
        if flattened_gradient[flat_index].item() != 0:
            return (
                parameter,
                flat_index,
                parameter.detach().flatten()[flat_index].clone(),
            )
    raise AssertionError("video-only backward produced no nonzero Wan gradients")


def assert_wan_update_probe_changed(probe) -> None:
    """Fail when the optimizer step leaves the captured Wan parameter unchanged."""
    parameter, flat_index, value_before = probe
    value_after = parameter.detach().flatten()[flat_index]
    if torch.equal(value_before, value_after):
        raise AssertionError("frozen-auxiliary optimizer did not update a Wan parameter")


def metric_device_mismatches(metrics, accelerator_device) -> dict[str, str]:
    """Return metrics whose device type cannot be reduced by this accelerator."""
    accelerator_device_type = torch.device(accelerator_device).type
    return {
        name: str(value.device)
        for name, value in metrics.items()
        if value.device.type != accelerator_device_type
    }


def main() -> None:
    """Exercise the production model and first real sample through one backward pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/joint_simgen_480_8gpu_video_only_3slot.yaml",
    )
    parser.add_argument("--expected-processes", type=int, default=1)
    parser.add_argument(
        "--frozen-auxiliary",
        action="store_true",
        help="freeze the PC and bridge modules, then run one Wan-only optimizer step",
    )
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
    optimizer = None
    if args.frozen_auxiliary:
        freeze_pc_and_bridges(model)
        optimizer = create_frozen_auxiliary_optimizer(model, config["optimizer"])
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
    else:
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
    if args.frozen_auxiliary:
        assert_frozen_pc_and_bridges_have_no_gradients(unwrapped)
        metrics = {
            "loss": loss.detach(),
            "bridge_gradient_norm": _bridge_gradient_norm(unwrapped).detach(),
            "video_gradient_norm": video_gradient_norm(unwrapped).detach(),
            "pc_gradient_norm": pc_gradient_norm(unwrapped).detach(),
        }
        incorrect_devices = metric_device_mismatches(metrics, accelerator.device)
        if incorrect_devices:
            raise AssertionError(
                "distributed metrics must be on the accelerator device: "
                f"expected={accelerator.device}, actual={incorrect_devices}"
            )
        reduced_metrics = {
            name: accelerator.reduce(value, reduction="mean")
            for name, value in metrics.items()
        }
        if metric_device_mismatches(reduced_metrics, accelerator.device):
            raise AssertionError(
                "metric reduction moved a value off the accelerator device"
            )
        update_probe = capture_wan_update_probe(unwrapped)
        accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
        optimizer.step()
        assert_wan_update_probe_changed(update_probe)
        optimizer.zero_grad(set_to_none=True)
    print(
        "PASS: "
        f"rank={accelerator.process_index} production-shaped video-only "
        f"{'frozen-auxiliary optimizer step' if args.frozen_auxiliary else 'backward'} completed",
        flush=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
