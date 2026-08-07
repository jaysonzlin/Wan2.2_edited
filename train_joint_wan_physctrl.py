"""Train base Wan 2.2 TI2V 5B and per-object PhysCtrl trajectories jointly."""

import argparse
import shutil
from pathlib import Path

import torch

from training.joint_config import load_joint_config


def create_joint_optimizer(model, optimizer_config: dict) -> torch.optim.AdamW:
    """Create independently configured AdamW groups for video, BCA, and PC modules."""
    groups = []
    for name, parameters in (
        ("video", model.wan_model.parameters()),
        ("bca", model.bridges.parameters()),
        ("pc", model.pc_model.parameters()),
    ):
        settings = optimizer_config[name]
        groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": settings["lr"],
                "betas": tuple(settings["betas"]),
                "eps": settings["eps"],
                "weight_decay": settings["weight_decay"],
            }
        )
    return torch.optim.AdamW(groups)


def combine_joint_losses(
    video_loss: torch.Tensor,
    object_losses: torch.Tensor,
    rigid_loss_sum: torch.Tensor,
    rigid_loss_weight: float,
    deform_loss_sum: torch.Tensor | None = None,
    deform_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Add video, x0, and optional per-object auxiliary losses without averaging objects."""
    total = video_loss + object_losses.sum() + rigid_loss_weight * rigid_loss_sum
    return (
        total
        if deform_loss_sum is None
        else total + deform_loss_weight * deform_loss_sum
    )


def per_object_metric_values(prefix: str, losses: torch.Tensor) -> dict[str, float]:
    """Name object-slot loss scalars consistently for tracker logging."""
    return {f"{prefix}_{index:03d}": value.item() for index, value in enumerate(losses)}


def rigid_loss_terms(
    enabled: bool,
    initial_point_clouds: torch.Tensor,
    prediction: torch.Tensor,
    *,
    neighbors: int,
    rigid_loss_fn,
) -> torch.Tensor:
    """Evaluate per-object rigidity only when the joint objective enables it."""
    if not enabled:
        return torch.zeros(
            initial_point_clouds.shape[:2],
            device=prediction.device,
            dtype=prediction.dtype,
        )
    return rigid_loss_fn(initial_point_clouds, prediction, neighbors=neighbors)


def add_rigid_metrics(
    metrics: dict[str, float],
    enabled: bool,
    rigid_losses: torch.Tensor,
) -> dict[str, float]:
    """Append rigid objective metrics only when that objective ran."""
    if enabled:
        metrics["train/rigid_loss_sum"] = rigid_losses.sum().detach().item()
        metrics.update(
            per_object_metric_values(
                "train/rigid_loss_object", rigid_losses[0].detach()
            )
        )
    return metrics


def deform_loss_terms(
    enabled: bool,
    initial_point_clouds: torch.Tensor,
    prediction: torch.Tensor,
    *,
    deform_f: torch.Tensor | None = None,
    deform_c: torch.Tensor | None = None,
    deform_volume: torch.Tensor | None = None,
    deform_baseline: torch.Tensor | None = None,
    deform_grid_origin: torch.Tensor | None = None,
    deform_grid_scale: torch.Tensor | None = None,
    deform_loss_fn=None,
) -> torch.Tensor:
    """Evaluate deformation supervision only after its optional HDF5 contract was loaded."""
    if not enabled:
        return torch.zeros(
            initial_point_clouds.shape[:2],
            device=prediction.device,
            dtype=prediction.dtype,
        )
    if any(
        value is None
        for value in (
            deform_f,
            deform_c,
            deform_volume,
            deform_baseline,
            deform_grid_origin,
            deform_grid_scale,
            deform_loss_fn,
        )
    ):
        raise ValueError(
            "enabled deform loss requires all deformation fields and an objective"
        )
    return deform_loss_fn(
        initial_point_clouds,
        prediction,
        deform_f=deform_f,
        deform_c=deform_c,
        deform_volume=deform_volume,
        deform_baseline=deform_baseline,
        deform_grid_origin=deform_grid_origin,
        deform_grid_scale=deform_grid_scale,
    )


def add_deform_metrics(
    metrics: dict[str, float], enabled: bool, deform_losses: torch.Tensor
) -> dict[str, float]:
    """Append deformation objective metrics only when that objective ran."""
    if enabled:
        metrics["train/deform_loss_sum"] = deform_losses.sum().detach().item()
        metrics.update(
            per_object_metric_values(
                "train/deform_loss_object", deform_losses[0].detach()
            )
        )
    return metrics


def should_save_joint_visualization(global_step: int, every_steps: int = 250) -> bool:
    if every_steps <= 0:
        raise ValueError("every_steps must be positive")
    return global_step > 0 and global_step % every_steps == 0


def should_log_denoised_latent_mse(global_step: int, every_steps: int = 50) -> bool:
    if every_steps <= 0:
        raise ValueError("every_steps must be positive")
    return global_step > 0 and global_step % every_steps == 0


def video_gradient_norm(model) -> torch.Tensor:
    """Return the pre-clip global L2 gradient norm for the Wan DiT only."""
    gradients = [
        parameter.grad.detach().norm()
        for parameter in model.wan_model.parameters()
        if parameter.grad is not None
    ]
    return torch.stack(gradients).norm() if gradients else torch.zeros(())


def create_progress_bar(total: int, initial: int, enabled: bool):
    from tqdm.auto import tqdm

    return tqdm(
        total=total,
        initial=initial,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not enabled,
    )


def _joint_checkpoint_paths(output_dir: Path, setting: str | None) -> list[Path]:
    if not setting:
        return []
    if setting != "latest":
        return [Path(setting)]
    return sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )


def load_joint_checkpoint_with_fallback(
    accelerator, output_dir: Path, setting: str | None
) -> Path | None:
    """Restore an explicit checkpoint or the newest complete checkpoint selected by ``latest``."""
    checkpoints = _joint_checkpoint_paths(output_dir, setting)
    if not checkpoints:
        return None
    failures = []
    for checkpoint in checkpoints:
        try:
            accelerator.load_state(checkpoint)
        except Exception as error:
            if setting != "latest":
                raise
            failures.append((checkpoint, error))
        else:
            return checkpoint
    attempted = ", ".join(path.name for path, _ in failures)
    raise RuntimeError(
        f"Could not load any checkpoint selected by resume_from_checkpoint=latest: {attempted}"
    ) from failures[-1][1]


def prune_joint_checkpoints(root: str | Path, limit: int) -> None:
    """Keep the newest ``limit`` numeric joint-training checkpoint directories."""
    checkpoints = sorted(
        (path for path in Path(root).glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    for checkpoint in checkpoints[:-limit]:
        shutil.rmtree(checkpoint)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _encode_videos(vae, videos: torch.Tensor) -> torch.Tensor:
    clips = [clip.permute(1, 0, 2, 3).contiguous() for clip in videos]
    with torch.no_grad():
        return torch.stack(vae.encode(clips))


def _bridge_gradient_norm(model) -> torch.Tensor:
    norms = [
        parameter.grad.detach().norm()
        for parameter in model.bridges.parameters()
        if parameter.grad is not None
    ]
    return (
        torch.stack(norms).norm()
        if norms
        else torch.zeros((), device=next(model.parameters()).device)
    )


def pc_gradient_norm(model) -> torch.Tensor:
    """Return the pre-clip global L2 gradient norm for the PhysCtrl model only."""
    gradients = [
        parameter.grad.detach().norm()
        for parameter in model.pc_model.parameters()
        if parameter.grad is not None
    ]
    return torch.stack(gradients).norm() if gradients else torch.zeros(())


@torch.no_grad()
def _save_joint_visualization(
    model,
    vae,
    context,
    clean_latents,
    point_clouds,
    linear,
    angular,
    output_dir,
    step,
    config,
    device,
    save_artifacts: bool,
) -> torch.Tensor:
    from diffusers import DDIMScheduler

    from train_i2v_832x480 import save_visualization
    from training.pc_visualization import save_pointcloud_comparison_mp4
    from wan.joint_pc_pipeline import JointWanPhysCtrlPipeline
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    objective = config["objective"]
    video_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=objective["num_train_timesteps"],
        prediction_type="flow_prediction",
        shift=1,
        use_dynamic_shifting=False,
    )
    pc_scheduler = DDIMScheduler(
        num_train_timesteps=objective["num_train_timesteps"],
        beta_schedule=objective["beta_schedule"],
        prediction_type="sample",
        clip_sample=False,
    )
    pipeline = JointWanPhysCtrlPipeline(
        model, video_scheduler, pc_scheduler, time_shift=objective["time_shift"]
    )
    initial = point_clouds[0, :, 0]
    sample = pipeline(
        condition_latent=clean_latents[0, :, :1],
        video_shape=tuple(clean_latents.shape[1:]),
        context=context,
        initial_point_clouds=initial,
        initial_linear_velocities=linear[0],
        initial_angular_velocities=angular[0],
        num_inference_steps=config["sampling"]["num_inference_steps"],
        generator=torch.Generator(device=device).manual_seed(
            config["training"]["seed"]
        ),
    )
    if not save_artifacts:
        return sample.video_latent
    sample_dir = Path(output_dir) / "visualizations" / f"step_{step:07d}"
    save_visualization(
        vae,
        sample.video_latent,
        sample_dir / "video.mp4",
        config["visualization"]["fps"],
    )
    predicted = torch.cat((initial.unsqueeze(1), sample.future_point_clouds[0]), dim=1)
    ground_truth = point_clouds[0]
    predicted = predicted.squeeze(2).permute(1, 0, 2, 3).cpu().numpy()
    ground_truth = ground_truth.squeeze(2).permute(1, 0, 2, 3).cpu().numpy()
    for object_index in range(predicted.shape[1]):
        save_pointcloud_comparison_mp4(
            predicted[:, object_index : object_index + 1],
            ground_truth[:, object_index : object_index + 1],
            sample_dir / f"object_{object_index:03d}_trajectory_comparison.mp4",
            config["visualization"]["fps"],
        )
    return sample.video_latent


def main(config: dict | None = None) -> None:
    """Launch distributed joint training with an empty prompt and a single optimizer."""
    if config is None:
        args = _parse_args()
        config = load_joint_config(args.config, args.overrides)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    import yaml
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader
    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )
    from training.joint_dataset import JointWanPhysCtrlDataset, joint_collate
    from training.joint_objectives import (
        make_aligned_multi_object_pc_ddpm_batch,
        per_object_baseline_corrected_deform_loss,
        per_object_pc_x0_mse,
        per_object_rigid_edge_length_loss,
    )
    from training.schedules import create_lr_scheduler
    from training.wan_i2v_training import (
        expand_latent_timesteps,
        denoised_latent_mse,
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
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.yaml").open("w") as handle:
            yaml.safe_dump(config, handle)
    if config["logging"].get("report_to"):
        accelerator.init_trackers(config["logging"]["project"], config=config)
    enable_deform_loss = config["objective"].get("enable_deform_loss", False)
    dataset = JointWanPhysCtrlDataset(
        config["data"]["dataset_root"],
        expected_size=(config["data"]["width"], config["data"]["height"]),
        expected_points=config["data"]["num_points"],
        load_deformation_fields=enable_deform_loss,
        expected_deform_neighbors=(
            config["objective"].get("deform_loss_neighbors", 32)
            if enable_deform_loss
            else None
        ),
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
        n_future_frames=48,
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
    enable_rigid_loss = config["objective"].get("enable_rigid_loss", False)
    global_step = 0
    resume_path = load_joint_checkpoint_with_fallback(
        accelerator, output_dir, config["training"].get("resume_from_checkpoint")
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
            point_clouds = batch["point_clouds"].to(accelerator.device)
            if point_clouds.shape[1] > config["data"]["max_objects_per_sample"]:
                raise ValueError(
                    "sample exceeds the configured initial object-count curriculum"
                )
            videos = batch["video"].to(accelerator.device)
            linear = batch["initial_linear_velocities"].to(accelerator.device)
            angular = batch["initial_angular_velocities"].to(accelerator.device)
            deform_fields = (
                {
                    "deform_f": batch["deform_F"].to(accelerator.device),
                    "deform_c": batch["deform_C"].to(accelerator.device),
                    "deform_volume": batch["deform_volume"].to(accelerator.device),
                    "deform_baseline": batch["deform_baseline"].to(accelerator.device),
                    "deform_grid_origin": batch["deform_grid_origin"].to(
                        accelerator.device
                    ),
                    "deform_grid_scale": batch["deform_grid_scale"].to(
                        accelerator.device
                    ),
                }
                if enable_deform_loss
                else None
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
                rigid_losses = rigid_loss_terms(
                    enable_rigid_loss,
                    point_clouds[:, :, 0],
                    pc_prediction,
                    neighbors=config["objective"].get("rigid_loss_neighbors", 16),
                    rigid_loss_fn=per_object_rigid_edge_length_loss,
                )
                rigid_loss_sum = rigid_losses.sum()
                deform_losses = deform_loss_terms(
                    enable_deform_loss,
                    point_clouds[:, :, 0],
                    pc_prediction,
                    **(deform_fields or {}),
                    deform_loss_fn=per_object_baseline_corrected_deform_loss,
                )
                deform_loss_sum = deform_losses.sum()
                loss = combine_joint_losses(
                    video_loss,
                    object_losses,
                    rigid_loss_sum=rigid_loss_sum,
                    rigid_loss_weight=config["objective"].get("rigid_loss_weight", 0.0),
                    deform_loss_sum=deform_loss_sum,
                    deform_loss_weight=config["objective"].get(
                        "deform_loss_weight", 0.001
                    ),
                )
                accelerator.backward(loss)
                bridge_grad_norm = _bridge_gradient_norm(
                    accelerator.unwrap_model(model)
                )
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
            add_rigid_metrics(metrics, enable_rigid_loss, rigid_losses)
            add_deform_metrics(metrics, enable_deform_loss, deform_losses)
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
