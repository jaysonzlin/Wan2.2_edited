"""Video-only noise ablation for history-conditioned joint SimGen training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from joint_simgen import (
    _create_training_accelerator,
    _encode_simgen_videos,
    _save_simgen_target_video,
    validation_video_paths,
)
from train_joint_wan_physctrl import (
    _bridge_gradient_norm,
    create_joint_optimizer,
    create_progress_bar,
    load_joint_checkpoint_with_fallback,
    pc_gradient_norm,
    prune_joint_checkpoints,
    video_gradient_norm,
)
from training.simgen_video_only_conditioning import (
    HISTORY_FRAMES,
    make_video_only_generators,
    make_sampling_conditioning,
    make_training_conditioning,
)
from training.simgen_video_only_config import load_simgen_video_only_config
from training.simgen_video_only_dataset import SimGenVideoDataset, simgen_video_collate
from training.wan_i2v_training import (
    expand_latent_timesteps,
    load_frozen_encoders,
    load_trainable_dit,
    make_flow_matching_batch,
    masked_velocity_mse,
)


def build_datasets(config: dict) -> tuple[SimGenVideoDataset, SimGenVideoDataset]:
    """Build the fixed video-only train and validation splits."""
    data = config["data"]
    train = SimGenVideoDataset(
        data["dataset_root"], list(range(data["train_start"], data["train_end"] + 1))
    )
    validation = SimGenVideoDataset(
        data["dataset_root"],
        list(range(data["validation_start"], data["validation_end"] + 1)),
    )
    return train, validation


def _create_video_only_training_accelerator(training: dict, logging: dict):
    """Allow the intentionally unsupervised final PC path in video-only DDP."""
    return _create_training_accelerator(
        training,
        logging,
        freeze_video=False,
        find_unused_parameters=True,
    )


def freeze_pc_and_bridges(model: torch.nn.Module) -> None:
    """Leave the random auxiliary branch structurally present but immutable."""
    model.pc_model.requires_grad_(False)
    model.bridges.requires_grad_(False)


def assert_frozen_pc_and_bridges(model: torch.nn.Module) -> None:
    """Fail if auxiliary parameters can participate in optimization."""
    if any(parameter.requires_grad for parameter in model.pc_model.parameters()):
        raise RuntimeError("frozen video-only training requires pc_model to be frozen")
    if any(parameter.requires_grad for parameter in model.bridges.parameters()):
        raise RuntimeError("frozen video-only training requires bridges to be frozen")


def assert_frozen_pc_and_bridges_have_no_gradients(model: torch.nn.Module) -> None:
    """Fail fast when an immutable auxiliary parameter receives a gradient."""
    unexpected = [
        parameter
        for module in (model.pc_model, model.bridges)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if unexpected:
        raise RuntimeError("frozen PC or bridge parameters received gradients")


def create_frozen_auxiliary_optimizer(model: torch.nn.Module, optimizer_config: dict):
    """Create the Wan-only AdamW optimizer for the frozen-auxiliary ablation."""
    assert_frozen_pc_and_bridges(model)
    settings = optimizer_config["video"]
    return torch.optim.AdamW(
        [
            {
                "name": "video",
                "params": model.wan_model.parameters(),
                "lr": settings["lr"],
                "betas": tuple(settings["betas"]),
                "eps": settings["eps"],
                "weight_decay": settings["weight_decay"],
            }
        ]
    )


def _video_only_losses(
    batch, model, vae, text_encoder, video_generator, device, data, model_config, objective,
    pc_generator=None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Compute flow loss while retaining only synthetic PC-side inputs."""
    videos = batch["video"].to(device, non_blocking=True)
    clean_latents = _encode_simgen_videos(vae, videos)
    flow = make_flow_matching_batch(
        clean_latents,
        video_generator,
        objective["time_shift"],
        objective["num_train_timesteps"],
        history_frames=HISTORY_FRAMES,
    )
    conditioning = make_training_conditioning(
        batch_size=videos.shape[0],
        point_count=data["num_points"],
        utonia_feature_dim=model_config["utonia_feature_dim"],
        num_train_timesteps=objective["num_train_timesteps"],
        device=device,
        dtype=clean_latents.dtype,
        generator=pc_generator if pc_generator is not None else video_generator,
    )
    context = text_encoder([""] * videos.shape[0], device)
    video_prediction, _ = model(
        video_x=[flow.model_input[index] for index in range(videos.shape[0])],
        video_t=expand_latent_timesteps(
            flow.latent_timesteps, clean_latents.shape[-2], clean_latents.shape[-1]
        ),
        context=context,
        seq_len=flow.latent_timesteps.shape[1]
        * (clean_latents.shape[-2] // 2)
        * (clean_latents.shape[-1] // 2),
        noisy_future_state=conditioning.noisy_future_state,
        frame_times=conditioning.frame_times,
        init_pc=conditioning.init_pc,
        utonia_features=conditioning.utonia_features,
        return_pc_output=False,
    )
    return (
        masked_velocity_mse(torch.stack(video_prediction), flow.velocity_target, flow.loss_mask),
        clean_latents,
        context,
    )


def _sample_video_only(
    *, pipeline, condition_latent, video_shape, context, point_count,
    utonia_feature_dim, device, dtype, video_generator, num_inference_steps,
    pc_generator=None, advance_pc_state=True,
):
    """Sample one video with noise conditioning fixed for the whole trajectory."""
    history, features = make_sampling_conditioning(
        point_count=point_count,
        utonia_feature_dim=utonia_feature_dim,
        device=device,
        dtype=dtype,
        generator=pc_generator if pc_generator is not None else video_generator,
    )
    return pipeline(
        condition_latent=condition_latent,
        video_shape=video_shape,
        context=context,
        initial_point_clouds=history,
        initial_linear_velocities=None,
        initial_angular_velocities=None,
        utonia_features=features,
        num_inference_steps=num_inference_steps,
        generator=video_generator,
        pc_generator=pc_generator,
        advance_pc_state=advance_pc_state,
    )


def _save_sample_zero_visualization(model, vae, text_encoder, batch, output_dir, step, config, device):
    """Write sample-0 video artifacts without inspecting PC sampler outputs."""
    from diffusers import DDIMScheduler
    from train_i2v_832x480 import save_visualization
    from wan.joint_pc_pipeline import JointWanPhysCtrlPipeline
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    objective = config["objective"]
    videos = batch["video"].to(device, non_blocking=True)
    clean_latents = _encode_simgen_videos(vae, videos)
    pipeline = JointWanPhysCtrlPipeline(
        model,
        FlowUniPCMultistepScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            prediction_type="flow_prediction", shift=1, use_dynamic_shifting=False,
        ),
        DDIMScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            beta_schedule=objective["beta_schedule"], prediction_type="sample", clip_sample=False,
        ),
        time_shift=objective["time_shift"],
    )
    video_generator, pc_generator = make_video_only_generators(
        device, config["training"]["seed"]
    )
    sample = _sample_video_only(
        pipeline=pipeline,
        condition_latent=clean_latents[0, :, :HISTORY_FRAMES],
        video_shape=tuple(clean_latents.shape[1:]),
        context=text_encoder([""], device),
        point_count=config["data"]["num_points"],
        utonia_feature_dim=config["model"]["utonia_feature_dim"],
        device=device,
        dtype=clean_latents.dtype,
        video_generator=video_generator,
        pc_generator=pc_generator,
        num_inference_steps=config["sampling"]["num_inference_steps"],
        advance_pc_state=not config["sampling"].get("freeze_auxiliary_pc_state", False),
    )
    destination = Path(output_dir) / "visualizations" / f"step_{step:07d}"
    save_visualization(vae, sample.video_latent, destination / "video.mp4", config["visualization"]["fps"])
    _save_simgen_target_video(videos[0], destination / "target.mp4", config["visualization"]["fps"])


@torch.inference_mode()
def visualize_validation(
    config: dict, checkpoint_path: str | Path, output_dir: str | Path
) -> None:
    """Render every video-only validation sample from one training checkpoint."""
    from diffusers import DDIMScheduler
    from safetensors.torch import load_file

    from train_i2v_832x480 import save_visualization
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.joint_pc_pipeline import JointWanPhysCtrlPipeline
    from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
    from wan.modules.pc_trajectory import PCTrajectoryModel
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("validation visualization requires a CUDA GPU")
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "model.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing model.safetensors: {checkpoint}")

    data, training, objective = (
        config["data"],
        config["training"],
        config["objective"],
    )
    validation_dataset = SimGenVideoDataset(
        data["dataset_root"],
        list(range(data["validation_start"], data["validation_end"] + 1)),
    )
    device = torch.device("cuda")
    vae, text_encoder = load_frozen_encoders(
        config["model"]["checkpoint_dir"], ti2v_5B, device
    )
    model = JointWanPhysCtrlModel(
        load_trainable_dit(
            config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"]
        ),
        PCTrajectoryModel(
            n_points=data["num_points"], n_future_frames=45, latent_dim=256,
            n_layers=8, num_heads=4, objective_type="ddpm", conditioning="history",
            history_frames=HISTORY_FRAMES,
            utonia_feature_dim=config["model"]["utonia_feature_dim"],
        ),
    ).to(device)
    state = load_file(str(state_path), device=str(device))
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    pipeline = JointWanPhysCtrlPipeline(
        model,
        FlowUniPCMultistepScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            prediction_type="flow_prediction", shift=1, use_dynamic_shifting=False,
        ),
        DDIMScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            beta_schedule=objective["beta_schedule"], prediction_type="sample",
            clip_sample=False,
        ),
        time_shift=objective["time_shift"],
    )
    destination = Path(output_dir)
    for index, validation_sample in enumerate(validation_dataset):
        batch = simgen_video_collate([validation_sample])
        videos = batch["video"].to(device, non_blocking=True)
        clean_latents = _encode_simgen_videos(vae, videos)
        video_generator, pc_generator = make_video_only_generators(
            device, training["seed"]
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            sample = _sample_video_only(
                pipeline=pipeline,
                condition_latent=clean_latents[0, :, :HISTORY_FRAMES],
                video_shape=tuple(clean_latents.shape[1:]),
                context=text_encoder([""], device),
                point_count=data["num_points"],
                utonia_feature_dim=config["model"]["utonia_feature_dim"],
                device=device,
                dtype=clean_latents.dtype,
                video_generator=video_generator,
                pc_generator=pc_generator,
                num_inference_steps=config["sampling"]["num_inference_steps"],
                advance_pc_state=not config["sampling"].get("freeze_auxiliary_pc_state", False),
            )
        prediction_path, target_path = validation_video_paths(
            destination, str(batch["sample_id"])
        )
        save_visualization(
            vae, sample.video_latent, prediction_path, config["visualization"]["fps"]
        )
        _save_simgen_target_video(
            videos[0], target_path, config["visualization"]["fps"]
        )
        print(
            f"Rendered validation sample {index + 1}/{len(validation_dataset)}: "
            f"{prediction_path}",
            flush=True,
        )


def run_training(config: dict) -> None:
    """Train the preserved joint architecture from video supervision only."""
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    import yaml

    from training.schedules import create_lr_scheduler
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
    from wan.modules.pc_trajectory import PCTrajectoryModel

    data, training, logging, objective = (
        config["data"], config["training"], config["logging"], config["objective"]
    )
    accelerator = _create_video_only_training_accelerator(training, logging)
    set_seed(training["seed"], device_specific=False)
    train_dataset, validation_dataset = build_datasets(config)
    visualization_batch = simgen_video_collate([train_dataset[0]])
    output_dir = Path(logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.yaml").open("w") as output:
            yaml.safe_dump(config, output)
    accelerator.wait_for_everyone()
    if logging.get("report_to"):
        accelerator.init_trackers(
            logging["project"], config=config, init_kwargs={"wandb": {"name": logging["run_name"]}}
        )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=data["dataloader_num_workers"], pin_memory=True,
        collate_fn=simgen_video_collate,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False,
        num_workers=data["dataloader_num_workers"], pin_memory=True,
        collate_fn=simgen_video_collate,
    )
    vae, text_encoder = load_frozen_encoders(
        config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device
    )
    model = JointWanPhysCtrlModel(
        load_trainable_dit(
            config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"]
        ),
        PCTrajectoryModel(
            n_points=data["num_points"], n_future_frames=45, latent_dim=256,
            n_layers=8, num_heads=4, objective_type="ddpm", conditioning="history",
            history_frames=HISTORY_FRAMES,
            utonia_feature_dim=config["model"]["utonia_feature_dim"],
        ),
    )
    frozen_auxiliary = training.get("freeze_pc_and_bridges", False)
    if frozen_auxiliary:
        freeze_pc_and_bridges(model)
        optimizer = create_frozen_auxiliary_optimizer(model, config["optimizer"])
    else:
        optimizer = create_joint_optimizer(model, config["optimizer"])
    lr_scheduler = create_lr_scheduler(
        training["lr_scheduler"], optimizer, training["warmup_steps"], training["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if frozen_auxiliary:
        assert_frozen_pc_and_bridges(accelerator.unwrap_model(model))
    resume_path = load_joint_checkpoint_with_fallback(
        accelerator, output_dir, training.get("resume_from_checkpoint")
    )
    set_seed(training["seed"], device_specific=False)
    video_generator, pc_generator = make_video_only_generators(
        accelerator.device, training["seed"]
    )
    validation_video_generator, validation_pc_generator = make_video_only_generators(
        accelerator.device, training["seed"]
    )
    global_step = int(resume_path.name.removeprefix("checkpoint-")) if resume_path else 0
    progress = create_progress_bar(training["max_train_steps"], global_step, accelerator.is_main_process)
    while global_step < training["max_train_steps"]:
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, _, _ = _video_only_losses(
                    batch, model, vae, text_encoder, video_generator, accelerator.device,
                    data, config["model"], objective, pc_generator,
                )
                accelerator.backward(loss)
                unwrapped = accelerator.unwrap_model(model)
                if frozen_auxiliary:
                    assert_frozen_pc_and_bridges_have_no_gradients(unwrapped)
                bridge_norm = _bridge_gradient_norm(unwrapped)
                wan_norm = video_gradient_norm(unwrapped)
                pc_norm = pc_gradient_norm(unwrapped)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            metrics = {
                "train/video_loss": accelerator.reduce(loss.detach(), reduction="mean").item(),
                "train/loss": accelerator.reduce(loss.detach(), reduction="mean").item(),
                "train/bridge_gradient_norm": accelerator.reduce(bridge_norm.detach(), reduction="mean").item(),
                "train/video_gradient_norm": accelerator.reduce(wan_norm.detach(), reduction="mean").item(),
                "train/pc_gradient_norm": accelerator.reduce(pc_norm.detach(), reduction="mean").item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/frozen_auxiliary_modules": float(frozen_auxiliary),
            }
            if accelerator.is_main_process:
                accelerator.log(metrics, step=global_step)
                progress.update(1)
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_joint_checkpoints(output_dir, training["checkpoints_total_limit"])
                accelerator.wait_for_everyone()
            if global_step % config["validation"]["every_steps"] == 0:
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    was_training = unwrapped.training
                    unwrapped.eval()
                    losses = []
                    with torch.no_grad():
                        for validation_batch in validation_loader:
                            validation_loss, _, _ = _video_only_losses(
                                validation_batch, unwrapped, vae, text_encoder, validation_video_generator,
                                accelerator.device, data, config["model"], objective, validation_pc_generator,
                            )
                            losses.append(validation_loss)
                    accelerator.log({"validation/video_loss": torch.stack(losses).mean().item(), "validation/loss": torch.stack(losses).mean().item()}, step=global_step)
                    _save_sample_zero_visualization(
                        unwrapped, vae, text_encoder, visualization_batch, output_dir,
                        global_step, config, accelerator.device,
                    )
                    if was_training:
                        unwrapped.train()
                accelerator.wait_for_everyone()
            if global_step >= training["max_train_steps"]:
                break
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir / "final_checkpoint")
    progress.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_training(load_simgen_video_only_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
