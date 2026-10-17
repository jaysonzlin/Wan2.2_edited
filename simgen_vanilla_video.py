"""Train the native Wan video path on SimGen without PC or bridge modules."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.simgen_vanilla_video_config import load_simgen_vanilla_video_config
from training.simgen_video_only_dataset import SimGenVideoDataset, simgen_video_collate
from training.wan_i2v_training import (
    expand_latent_timesteps,
    make_flow_matching_batch,
    masked_velocity_mse,
)


HISTORY_FRAMES = 4


def _encode_simgen_videos(vae, videos: torch.Tensor) -> torch.Tensor:
    """Encode native ``[B, T, C, H, W]`` videos to Wan latents."""
    clips = [clip.permute(1, 0, 2, 3).contiguous() for clip in videos]
    with torch.no_grad():
        return torch.stack(vae.encode(clips))


def build_datasets(config: dict) -> tuple[SimGenVideoDataset, SimGenVideoDataset]:
    """Construct the controlled train and held-out video splits."""
    data = config["data"]
    return (
        SimGenVideoDataset(data["dataset_root"], list(range(data["train_start"], data["train_end"] + 1))),
        SimGenVideoDataset(
            data["dataset_root"], list(range(data["validation_start"], data["validation_end"] + 1))
        ),
    )


def create_vanilla_optimizer(model: torch.nn.Module, settings: dict) -> torch.optim.AdamW:
    """Create the experiment's one-group Wan optimizer."""
    return torch.optim.AdamW(
        [{
            "name": "video",
            "params": model.parameters(),
            "lr": settings["lr"],
            "betas": tuple(settings["betas"]),
            "eps": settings["eps"],
            "weight_decay": settings["weight_decay"],
        }]
    )


def vanilla_video_loss(
    batch, model, vae, text_encoder, generator: torch.Generator, device,
    objective: dict, history_frames: int = HISTORY_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the native Wan masked flow loss without auxiliary tensors."""
    videos = batch["video"].to(device, non_blocking=True)
    clean_latents = _encode_simgen_videos(vae, videos)
    flow = make_flow_matching_batch(
        clean_latents,
        generator,
        objective["time_shift"],
        objective["num_train_timesteps"],
        history_frames=history_frames,
    )
    prediction = model(
        [flow.model_input[index] for index in range(videos.shape[0])],
        expand_latent_timesteps(
            flow.latent_timesteps, clean_latents.shape[-2], clean_latents.shape[-1]
        ),
        text_encoder([""] * videos.shape[0], device),
        flow.latent_timesteps.shape[1]
        * (clean_latents.shape[-2] // 2)
        * (clean_latents.shape[-1] // 2),
    )
    return masked_velocity_mse(torch.stack(prediction), flow.velocity_target, flow.loss_mask), clean_latents


def _video_scheduler(config: dict):
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    objective = config["objective"]
    return FlowUniPCMultistepScheduler(
        num_train_timesteps=objective["num_train_timesteps"],
        prediction_type="flow_prediction",
        shift=1,
        use_dynamic_shifting=False,
    )


@torch.inference_mode()
def render_video_sample(*, model, vae, text_encoder, batch, output_dir: str | Path, config: dict, device, generator: torch.Generator) -> None:
    """Render one deterministic prediction and its native RGB target."""
    from joint_simgen import _save_simgen_target_video, validation_video_paths
    from train_i2v_832x480 import save_visualization
    from wan.video_pipeline import WanVideoPipeline

    videos = batch["video"].to(device, non_blocking=True)
    clean_latents = _encode_simgen_videos(vae, videos)
    sample = WanVideoPipeline(model, _video_scheduler(config), config["objective"]["time_shift"])(
        condition_latent=clean_latents[0, :, :HISTORY_FRAMES],
        video_shape=tuple(clean_latents.shape[1:]),
        context=text_encoder([""], device),
        num_inference_steps=config["sampling"]["num_inference_steps"],
        generator=generator,
    )
    prediction_path, target_path = validation_video_paths(output_dir, str(batch["sample_id"]))
    save_visualization(vae, sample.video_latent, prediction_path, config["visualization"]["fps"])
    _save_simgen_target_video(videos[0], target_path, config["visualization"]["fps"])


@torch.inference_mode()
def render_validation_videos(*, model, vae, text_encoder, validation_dataset, output_dir: str | Path, config: dict, device) -> None:
    """Render all held-out samples with fixed, distinct per-sample seeds."""
    for index, validation_sample in enumerate(validation_dataset):
        render_video_sample(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            batch=simgen_video_collate([validation_sample]),
            output_dir=Path(output_dir) / "final_validation",
            config=config,
            device=device,
            generator=torch.Generator(device=device).manual_seed(config["training"]["seed"] + index),
        )


def _gradient_norm(model: torch.nn.Module) -> torch.Tensor:
    gradients = [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
    return torch.stack(gradients).norm() if gradients else next(model.parameters()).new_zeros(())


def training_metrics(accelerator, loss: torch.Tensor, gradient_norm: torch.Tensor, learning_rate: float) -> dict[str, float]:
    """Reduce per-rank video metrics before rank zero writes them."""
    reduced_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
    reduced_gradient_norm = accelerator.reduce(gradient_norm.detach(), reduction="mean").item()
    return {
        "train/video_loss": reduced_loss,
        "train/loss": reduced_loss,
        "train/video_gradient_norm": reduced_gradient_norm,
        "train/learning_rate": learning_rate,
    }


def run_training(config: dict) -> None:
    """Run the fresh base-Wan, no-PC/bridge training experiment."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    import yaml

    from training.schedules import create_lr_scheduler
    from training.wan_i2v_training import load_frozen_encoders, load_trainable_dit
    from wan.configs.wan_ti2v_5B import ti2v_5B

    data, training, logging, objective = (
        config["data"], config["training"], config["logging"], config["objective"]
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"],
        log_with=logging.get("report_to") or None,
    )
    set_seed(training["seed"], device_specific=False)
    train_dataset, validation_dataset = build_datasets(config)
    visualization_batch = simgen_video_collate([train_dataset[0]])
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=data["dataloader_num_workers"],
        pin_memory=True, collate_fn=simgen_video_collate,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=data["dataloader_num_workers"],
        pin_memory=True, collate_fn=simgen_video_collate,
    )
    output_dir = Path(logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.yaml").open("w") as handle:
            yaml.safe_dump(config, handle)
    if logging.get("report_to"):
        accelerator.init_trackers(logging["project"], config=config)

    vae, text_encoder = load_frozen_encoders(config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device)
    model = load_trainable_dit(config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"])
    optimizer = create_vanilla_optimizer(model, config["optimizer"]["video"])
    lr_scheduler = create_lr_scheduler(
        training["lr_scheduler"], optimizer, training["warmup_steps"], training["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup, constant_factory=get_constant_schedule_with_warmup,
    )
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(training["seed"])

    global_step = 0
    while global_step < training["max_train_steps"]:
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, _ = vanilla_video_loss(batch, model, vae, text_encoder, generator, accelerator.device, objective)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                gradient_norm = _gradient_norm(accelerator.unwrap_model(model))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            metrics = training_metrics(
                accelerator, loss, gradient_norm, lr_scheduler.get_last_lr()[0]
            )
            if accelerator.is_main_process:
                accelerator.log(metrics, step=global_step)
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
            if global_step % config["validation"]["every_steps"] == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                was_training = unwrapped.training
                unwrapped.eval()
                validation_generator = torch.Generator(device=accelerator.device).manual_seed(training["seed"])
                validation_losses = [
                    vanilla_video_loss(validation_batch, unwrapped, vae, text_encoder, validation_generator, accelerator.device, objective)[0]
                    for validation_batch in validation_loader
                ]
                mean_validation_loss = torch.stack(validation_losses).mean().item()
                accelerator.log({"validation/video_loss": mean_validation_loss, "validation/loss": mean_validation_loss}, step=global_step)
                render_video_sample(
                    model=unwrapped, vae=vae, text_encoder=text_encoder, batch=visualization_batch,
                    output_dir=output_dir / "visualizations" / f"step_{global_step:07d}", config=config,
                    device=accelerator.device, generator=torch.Generator(device=accelerator.device).manual_seed(training["seed"]),
                )
                if was_training:
                    unwrapped.train()
            accelerator.wait_for_everyone()
            if global_step >= training["max_train_steps"]:
                break
    accelerator.save_state(output_dir / "final_checkpoint")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        render_validation_videos(
            model=unwrapped, vae=vae, text_encoder=text_encoder, validation_dataset=validation_dataset,
            output_dir=output_dir, config=config, device=accelerator.device,
        )
        if was_training:
            unwrapped.train()
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_training(load_simgen_vanilla_video_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
