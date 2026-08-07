"""Train joint Wan--PhysCtrl models on one fixed temporal trajectory window."""

import argparse
from pathlib import Path

import torch
import yaml

from train_joint_wan_physctrl import (
    _bridge_gradient_norm,
    _encode_videos,
    _save_joint_visualization,
    create_joint_optimizer,
    create_progress_bar,
    load_joint_checkpoint_with_fallback,
    pc_gradient_norm,
    per_object_metric_values,
    prune_joint_checkpoints,
    should_log_denoised_latent_mse,
    should_save_joint_visualization,
    video_gradient_norm,
)
from training.trajectory_config import load_trajectory_joint_config
from training.trajectory_window import TrajectoryWindow, window_joint_tensors


def configured_future_frames(config: dict) -> int:
    """Return the fixed PC prediction horizon selected for this experiment."""
    return config["trajectory"]["future_frames"]


def assert_resume_window_matches(
    output_dir: Path, window: TrajectoryWindow, setting: str | None
) -> None:
    """Reject a requested resume when its saved experiment used another window."""
    if setting is None:
        return
    config_path = output_dir / "config.yaml"
    if not config_path.is_file():
        if setting == "latest":
            return
        raise ValueError("resume checkpoint is missing its parent config.yaml")
    with config_path.open() as handle:
        saved = yaml.safe_load(handle) or {}
    saved_trajectory = saved.get("trajectory")
    expected = {
        "start_frame": window.start_frame,
        "future_frames": window.future_frames,
    }
    if saved_trajectory != expected:
        raise ValueError("resume checkpoint was created for a different trajectory window")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main(config: dict | None = None) -> None:
    """Launch joint training on aligned video and point-cloud trajectory windows."""
    if config is None:
        args = _parse_args()
        config = load_trajectory_joint_config(args.config, args.overrides)

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    from training.joint_dataset import JointWanPhysCtrlDataset, joint_collate
    from training.joint_objectives import (
        make_aligned_multi_object_pc_ddpm_batch,
        per_object_pc_x0_mse,
    )
    from training.schedules import create_lr_scheduler
    from training.wan_i2v_training import (
        denoised_latent_mse,
        expand_latent_timesteps,
        load_frozen_encoders,
        load_trainable_dit,
        make_flow_matching_batch,
        masked_velocity_mse,
    )
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
    from wan.modules.pc_trajectory import PCTrajectoryModel

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["logging"].get("report_to") or None,
    )
    set_seed(config["training"]["seed"])
    window = TrajectoryWindow(**config["trajectory"])
    output_dir = Path(config["logging"]["output_dir"])
    resume_setting = config["training"].get("resume_from_checkpoint")
    assert_resume_window_matches(output_dir, window, resume_setting)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.yaml").open("w") as handle:
            yaml.safe_dump(config, handle)
    if config["logging"].get("report_to"):
        accelerator.init_trackers(config["logging"]["project"], config=config)

    dataset = JointWanPhysCtrlDataset(
        config["data"]["dataset_root"],
        expected_frames=config["data"]["num_frames"],
        expected_size=(config["data"]["width"], config["data"]["height"]),
        expected_points=config["data"]["num_points"],
        load_deformation_fields=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=joint_collate,
    )
    vae, text_encoder = load_frozen_encoders(
        config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device
    )
    wan_model = load_trainable_dit(
        config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"]
    )
    pc_model = PCTrajectoryModel(
        n_points=config["data"]["num_points"],
        n_future_frames=configured_future_frames(config),
        latent_dim=256,
        n_layers=8,
        num_heads=4,
        objective_type="ddpm",
    )
    model = JointWanPhysCtrlModel(wan_model, pc_model)
    optimizer = create_joint_optimizer(model, config["optimizer"])
    lr_scheduler = create_lr_scheduler(
        config["training"]["lr_scheduler"],
        optimizer,
        config["training"]["warmup_steps"],
        config["training"]["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["objective"]["num_train_timesteps"],
        beta_schedule=config["objective"]["beta_schedule"],
        prediction_type="sample",
        clip_sample=False,
    )
    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(
        config["training"]["seed"]
    )
    global_step = 0
    resume_path = load_joint_checkpoint_with_fallback(
        accelerator, output_dir, resume_setting
    )
    if resume_path:
        global_step = int(resume_path.name.removeprefix("checkpoint-"))
    progress_bar = create_progress_bar(
        total=config["training"]["max_train_steps"],
        initial=global_step,
        enabled=accelerator.is_main_process,
    )

    while global_step < config["training"]["max_train_steps"]:
        for batch in loader:
            raw_points = batch["point_clouds"].to(accelerator.device)
            if raw_points.shape[1] > config["data"]["max_objects_per_sample"]:
                raise ValueError(
                    "sample exceeds the configured initial object-count curriculum"
                )
            raw_videos = batch["video"].to(accelerator.device)
            videos, point_clouds, linear, angular = window_joint_tensors(
                raw_videos, raw_points, window
            )
            with accelerator.accumulate(model):
                clean_latents = _encode_videos(vae, videos)
                context = text_encoder([""], accelerator.device)
                flow = make_flow_matching_batch(
                    clean_latents,
                    generator,
                    config["objective"]["time_shift"],
                    config["objective"]["num_train_timesteps"],
                )
                pc_batch = make_aligned_multi_object_pc_ddpm_batch(
                    point_clouds, noise_scheduler, generator
                )
                video_prediction, pc_prediction = model(
                    video_x=[flow.model_input[0]],
                    video_t=expand_latent_timesteps(
                        flow.latent_timesteps,
                        clean_latents.shape[-2],
                        clean_latents.shape[-1],
                    ),
                    context=context,
                    seq_len=flow.latent_timesteps.shape[1]
                    * (clean_latents.shape[-2] // 2)
                    * (clean_latents.shape[-1] // 2),
                    noisy_future_state=pc_batch.model_input,
                    frame_times=pc_batch.frame_times,
                    init_pc=point_clouds[:, :, 0],
                    initial_linear_velocity=linear,
                    initial_angular_velocity=angular,
                )
                video_loss = masked_velocity_mse(
                    torch.stack(video_prediction), flow.velocity_target, flow.loss_mask
                )
                object_losses, pc_loss_sum = per_object_pc_x0_mse(
                    pc_prediction, pc_batch.target
                )
                loss = video_loss + pc_loss_sum
                accelerator.backward(loss)
                bridge_grad_norm = _bridge_gradient_norm(accelerator.unwrap_model(model))
                wan_grad_norm = video_gradient_norm(accelerator.unwrap_model(model))
                pc_grad_norm = pc_gradient_norm(accelerator.unwrap_model(model))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config["training"]["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            metrics = {
                "train/video_loss": video_loss.detach().item(),
                "train/pc_loss_sum": pc_loss_sum.detach().item(),
                "train/loss": loss.detach().item(),
                "train/bridge_gradient_norm": bridge_grad_norm.detach().item(),
                "train/video_gradient_norm": wan_grad_norm.detach().item(),
                "train/pc_gradient_norm": pc_grad_norm.detach().item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
            }
            metrics.update(
                per_object_metric_values(
                    "train/pc_loss_object", object_losses[0].detach()
                )
            )
            accelerator.log(metrics, step=global_step)
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )
            if global_step % config["training"]["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_joint_checkpoints(
                        output_dir, config["training"]["checkpoints_total_limit"]
                    )
            should_log_mse = should_log_denoised_latent_mse(
                global_step, config["training"]["denoised_latent_mse_every_steps"]
            )
            should_save_visualization = should_save_joint_visualization(
                global_step, config["visualization"]["every_steps"]
            )
            if accelerator.is_main_process and (
                should_log_mse or should_save_visualization
            ):
                unwrapped = accelerator.unwrap_model(model)
                was_training = unwrapped.training
                unwrapped.eval()
                sampled_latent = _save_joint_visualization(
                    unwrapped,
                    vae,
                    context,
                    clean_latents,
                    point_clouds,
                    linear,
                    angular,
                    output_dir,
                    global_step,
                    config,
                    accelerator.device,
                    should_save_visualization,
                )
                if should_log_mse:
                    accelerator.log(
                        {
                            "train/denoised_latent_mse": denoised_latent_mse(
                                sampled_latent, clean_latents[0]
                            ).item()
                        },
                        step=global_step,
                    )
                if was_training:
                    unwrapped.train()
            if global_step >= config["training"]["max_train_steps"]:
                break
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir / "final_checkpoint")
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
