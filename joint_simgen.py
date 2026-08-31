"""History-conditioned joint Wan/PhysCtrl training on native SimGen samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from train_joint_wan_physctrl import (
    _bridge_gradient_norm,
    _encode_videos,
    create_joint_optimizer,
    create_progress_bar,
    load_joint_checkpoint_with_fallback,
    pc_gradient_norm,
    per_object_metric_values,
    prune_joint_checkpoints,
    video_gradient_norm,
)
from training.simgen_joint_config import load_simgen_joint_config
from training.simgen_joint_dataset import SimGenJointDataset, simgen_joint_collate
from training.simgen_utonia_features import SimGenUtoniaCache, prepare_simgen_utonia_cache
from training.utonia_features import UtoniaFeatureExtractor


def prepare_cache(config: dict) -> int:
    """Create the three explicit canonical sample_0 feature records."""
    data = config["data"]
    root = Path(data["dataset_root"])
    metadata_path = root / "sample_0" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        instances = metadata.get("instances", metadata.get("objects"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read canonical SimGen metadata: {metadata_path}") from error
    if not isinstance(instances, list):
        raise ValueError(f"{metadata_path}: metadata must contain an instances list")
    ordinals = {}
    for entry in instances:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("id"), str):
            raise ValueError(f"{metadata_path}: every canonical instance needs string id and name")
        if entry["name"] in ordinals:
            raise ValueError(f"{metadata_path}: duplicate canonical class {entry['name']}")
        ordinals[entry["name"]] = entry["id"]
    if set(ordinals) != {"panda", "ball", "can"}:
        raise ValueError(f"{metadata_path}: sample_0 must declare exactly panda, ball, and can")
    extractor = UtoniaFeatureExtractor(
        data["utonia_cache_root"], config.get("training", {}).get("seed", 0)
    )
    return prepare_simgen_utonia_cache(
        {
            name: root / "sample_0" / "objects" / ordinal / "pc.hdf5"
            for name, ordinal in ordinals.items()
        },
        data["utonia_cache_root"],
        extractor,
    )


def _cache_preparation_world_size() -> int:
    """Return the process count requested by an Accelerate/torch launch."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def _shared_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Create the requested same-seed diffusion generator for one process."""
    return torch.Generator(device=device).manual_seed(seed)


def _reduced_mean(accelerator, local_sum: torch.Tensor, local_count: int) -> torch.Tensor:
    """Reduce a sum/count pair without weighting ranks equally."""
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(
        local_sum.new_tensor(local_count), reduction="sum"
    )
    return global_sum / global_count


def _validation_metrics(
    validation_losses: list[torch.Tensor], validation_pc_losses: list[torch.Tensor]
) -> dict[str, float]:
    """Return rank-zero means for the total and point-cloud validation losses."""
    return {
        "validation/loss": torch.stack(validation_losses).mean().item(),
        "validation/pc_loss_sum": torch.stack(validation_pc_losses).mean().item(),
    }


def _validation_loss_components(loss_outputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract detached total and point-cloud losses from one validation forward."""
    return loss_outputs[0].detach(), loss_outputs[2].detach()


def _visualization_history(point_clouds: torch.Tensor) -> torch.Tensor:
    """Return the object-major four-frame history expected by the sampler."""
    return point_clouds[0, :, :4].squeeze(2)


def build_datasets(config: dict) -> tuple[SimGenJointDataset, SimGenJointDataset]:
    """Build the fixed 490/10 split before model initialization."""
    data = config["data"]
    common = {
        "expected_points": data["num_points"],
        "utonia_cache_root": data["utonia_cache_root"],
    }
    train = SimGenJointDataset(
        data["dataset_root"], list(range(data["train_start"], data["train_end"] + 1)), **common
    )
    validation = SimGenJointDataset(
        data["dataset_root"],
        list(range(data["validation_start"], data["validation_end"] + 1)),
        **common,
    )
    return train, validation


def _encode_simgen_videos(vae, videos: torch.Tensor) -> torch.Tensor:
    """Encode ``[B, T, C, H, W]`` native RGB batches for Wan flow matching."""
    return _encode_videos(vae, videos)


def _make_multi_object_ddpm_batch(point_clouds, scheduler, generator):
    """DDPM-noise each object's 45-frame future while retaining the B/K layout."""
    from training.pc_ddpm import PCDDPMBatch, make_pc_ddpm_batch

    batch_size, object_count = point_clouds.shape[:2]
    flat = make_pc_ddpm_batch(
        point_clouds[:, :, 4:].flatten(0, 1), scheduler, generator, known_frames=4
    )
    return PCDDPMBatch(
        model_input=flat.model_input.unflatten(0, (batch_size, object_count)),
        target=flat.target.unflatten(0, (batch_size, object_count)),
        frame_times=flat.frame_times.unflatten(0, (batch_size, object_count)),
        timesteps=flat.timesteps.unflatten(0, (batch_size, object_count)),
    )


def _joint_simgen_losses(
    batch, model, vae, text_encoder, noise_scheduler, generator, device, objective
):
    """Build the four-history flow/DDPM objectives without velocity auxiliaries."""
    from training.joint_objectives import per_object_pc_x0_mse
    from training.wan_i2v_training import (
        expand_latent_timesteps,
        make_flow_matching_batch,
        masked_velocity_mse,
    )

    videos = batch["video"].to(device, non_blocking=True)
    point_clouds = batch["point_clouds"].to(device, non_blocking=True)
    utonia_features = batch["utonia_features"].to(device, non_blocking=True)
    clean_latents = _encode_simgen_videos(vae, videos)
    flow = make_flow_matching_batch(
        clean_latents,
        generator,
        objective["time_shift"],
        objective["num_train_timesteps"],
        history_frames=4,
    )
    pc_batch = _make_multi_object_ddpm_batch(point_clouds, noise_scheduler, generator)
    context = text_encoder([""] * videos.shape[0], device)
    video_prediction, pc_prediction = model(
        video_x=[flow.model_input[index] for index in range(videos.shape[0])],
        video_t=expand_latent_timesteps(
            flow.latent_timesteps, clean_latents.shape[-2], clean_latents.shape[-1]
        ),
        context=context,
        seq_len=flow.latent_timesteps.shape[1]
        * (clean_latents.shape[-2] // 2)
        * (clean_latents.shape[-1] // 2),
        noisy_future_state=pc_batch.model_input,
        frame_times=pc_batch.frame_times,
        init_pc=point_clouds[:, :, :4],
        utonia_features=utonia_features,
    )
    video_loss = masked_velocity_mse(
        torch.stack(video_prediction), flow.velocity_target, flow.loss_mask
    )
    object_losses, pc_loss_sum = per_object_pc_x0_mse(pc_prediction, pc_batch.target)
    return (
        video_loss + pc_loss_sum,
        video_loss,
        pc_loss_sum,
        object_losses,
        clean_latents,
        context,
        point_clouds,
        utonia_features,
    )


@torch.no_grad()
def _save_simgen_visualization(
    model, vae, context, clean_latents, point_clouds, utonia_features,
    output_dir, step, config, device,
) -> None:
    """Render the deterministic sample_0 history/future comparison at a checkpoint step."""
    from diffusers import DDIMScheduler

    from train_i2v_832x480 import save_visualization
    from training.pc_visualization import save_pointcloud_comparison_mp4
    from wan.joint_pc_pipeline import JointWanPhysCtrlPipeline
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    objective = config["objective"]
    pipeline = JointWanPhysCtrlPipeline(
        model,
        FlowUniPCMultistepScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            prediction_type="flow_prediction",
            shift=1,
            use_dynamic_shifting=False,
        ),
        DDIMScheduler(
            num_train_timesteps=objective["num_train_timesteps"],
            beta_schedule=objective["beta_schedule"],
            prediction_type="sample",
            clip_sample=False,
        ),
        time_shift=objective["time_shift"],
    )
    history = _visualization_history(point_clouds)
    sample = pipeline(
        condition_latent=clean_latents[0, :, :4],
        video_shape=tuple(clean_latents.shape[1:]),
        context=context,
        initial_point_clouds=history,
        initial_linear_velocities=None,
        initial_angular_velocities=None,
        utonia_features=utonia_features[0],
        num_inference_steps=config["sampling"]["num_inference_steps"],
        generator=torch.Generator(device=device).manual_seed(config["training"]["seed"]),
    )
    sample_dir = Path(output_dir) / "visualizations" / f"step_{step:07d}"
    save_visualization(vae, sample.video_latent, sample_dir / "video.mp4", config["visualization"]["fps"])
    predicted = torch.cat((point_clouds[0, :, :4], sample.future_point_clouds[0]), dim=1)
    for object_index in range(predicted.shape[0]):
        save_pointcloud_comparison_mp4(
            predicted[:, :, 0].permute(1, 0, 2, 3).cpu().numpy()[:, object_index : object_index + 1],
            point_clouds[0, :, :, 0].permute(1, 0, 2, 3).cpu().numpy()[:, object_index : object_index + 1],
            sample_dir / f"object_{object_index:03d}_trajectory_comparison.mp4",
            config["visualization"]["fps"],
        )


def run_training(config: dict) -> None:
    """Train the fixed 490/10 SimGen experiment without mutating its Utonia cache."""
    data = config["data"]
    # This explicit read-only construction is the preparation guard: normal training
    # never invokes the writer and fails before any loader/optimizer setup if absent.
    SimGenUtoniaCache(data["utonia_cache_root"])

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    import yaml

    from training.schedules import create_lr_scheduler
    from training.wan_i2v_training import load_frozen_encoders, load_trainable_dit
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.modules.joint_wan_physctrl import JointWanPhysCtrlModel
    from wan.modules.pc_trajectory import PCTrajectoryModel

    training, logging = config["training"], config["logging"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"],
        log_with=logging.get("report_to") or None,
    )
    set_seed(training["seed"], device_specific=False)
    train_dataset, validation_dataset = build_datasets(config)
    visualization_batch = simgen_joint_collate([train_dataset[0]])
    feature_width = visualization_batch["utonia_features"].shape[-1]
    output_dir = Path(logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.yaml").open("w") as output:
            yaml.safe_dump(config, output)
    accelerator.wait_for_everyone()
    if logging.get("report_to"):
        accelerator.init_trackers(logging["project"], config=config)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=data["dataloader_num_workers"], pin_memory=True,
        collate_fn=simgen_joint_collate,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False,
        num_workers=data["dataloader_num_workers"], pin_memory=True,
        collate_fn=simgen_joint_collate,
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
            history_frames=4, utonia_feature_dim=feature_width,
        ),
    )
    optimizer = create_joint_optimizer(model, config["optimizer"])
    lr_scheduler = create_lr_scheduler(
        training["lr_scheduler"], optimizer, training["warmup_steps"], training["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["objective"]["num_train_timesteps"],
        beta_schedule=config["objective"]["beta_schedule"], prediction_type="sample", clip_sample=False,
    )
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    resume_path = load_joint_checkpoint_with_fallback(
        accelerator, output_dir, training.get("resume_from_checkpoint")
    )
    # Cross-world-size resumes cannot recover missing rank-local RNG snapshots.
    # The requested contract is therefore a shared configured seed on every rank.
    set_seed(training["seed"], device_specific=False)
    generator = _shared_generator(accelerator.device, training["seed"])
    validation_generator = _shared_generator(accelerator.device, training["seed"])
    global_step = int(resume_path.name.removeprefix("checkpoint-")) if resume_path else 0
    progress_bar = create_progress_bar(training["max_train_steps"], global_step, accelerator.is_main_process)

    while global_step < training["max_train_steps"]:
        for batch in train_loader:
            if batch["point_clouds"].shape[1] > data["max_objects_per_sample"]:
                raise ValueError("sample exceeds the configured initial object-count curriculum")
            with accelerator.accumulate(model):
                (
                    loss, video_loss, pc_loss_sum, object_losses, clean_latents,
                    context, point_clouds, utonia_features,
                ) = _joint_simgen_losses(
                    batch, model, vae, text_encoder, noise_scheduler, generator,
                    accelerator.device, config["objective"],
                )
                accelerator.backward(loss)
                unwrapped = accelerator.unwrap_model(model)
                bridge_grad_norm = _bridge_gradient_norm(unwrapped)
                wan_grad_norm = video_gradient_norm(unwrapped)
                pc_grad_norm = pc_gradient_norm(unwrapped)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            reduced_metrics = {
                "train/video_loss": _reduced_mean(accelerator, video_loss.detach(), 1),
                "train/pc_loss_sum": _reduced_mean(accelerator, pc_loss_sum.detach(), 1),
                "train/loss": _reduced_mean(accelerator, loss.detach(), 1),
                "train/bridge_gradient_norm": _reduced_mean(accelerator, bridge_grad_norm.detach(), 1),
                "train/video_gradient_norm": _reduced_mean(accelerator, wan_grad_norm.detach(), 1),
                "train/pc_gradient_norm": _reduced_mean(accelerator, pc_grad_norm.detach(), 1),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
            }
            if accelerator.is_main_process:
                metrics = {
                    name: value.item() if isinstance(value, torch.Tensor) else value
                    for name, value in reduced_metrics.items()
                }
                if accelerator.num_processes == 1:
                    metrics.update(per_object_metric_values("train/pc_loss_object", object_losses[0].detach()))
                accelerator.log(metrics, step=global_step)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{metrics['train/loss']:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_joint_checkpoints(output_dir, training["checkpoints_total_limit"])
                accelerator.wait_for_everyone()
            if global_step % config["validation"]["every_steps"] == 0:
                unwrapped = accelerator.unwrap_model(model)
                was_training = unwrapped.training
                unwrapped.eval()
                if accelerator.is_main_process:
                    from tqdm.auto import tqdm

                    validation_losses = []
                    validation_pc_losses = []
                    with torch.no_grad():
                        validation_progress = tqdm(
                            validation_loader,
                            desc="Validation",
                            unit="batch",
                            dynamic_ncols=True,
                        )
                        for validation_batch in validation_progress:
                            validation_outputs = _joint_simgen_losses(
                                validation_batch, unwrapped, vae, text_encoder,
                                noise_scheduler, validation_generator, accelerator.device,
                                config["objective"],
                            )
                            validation_loss, validation_pc_loss = _validation_loss_components(
                                validation_outputs
                            )
                            validation_losses.append(validation_loss)
                            validation_pc_losses.append(validation_pc_loss)
                    accelerator.log(
                        _validation_metrics(validation_losses, validation_pc_losses),
                        step=global_step,
                    )
                    with torch.no_grad():
                        (
                            _, _, _, _, visual_latents, visual_context, visual_points, visual_features,
                        ) = _joint_simgen_losses(
                            visualization_batch, unwrapped, vae, text_encoder, noise_scheduler,
                            validation_generator, accelerator.device, config["objective"],
                        )
                    _save_simgen_visualization(
                        unwrapped, vae, visual_context, visual_latents, visual_points, visual_features,
                        output_dir, global_step, config, accelerator.device,
                    )
                accelerator.wait_for_everyone()
                if was_training:
                    unwrapped.train()
            if global_step >= training["max_train_steps"]:
                break
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir / "final_checkpoint")
    progress_bar.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-utonia-cache", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_simgen_joint_config(args.config, args.overrides)
    if args.prepare_utonia_cache:
        if _cache_preparation_world_size() > 1:
            raise ValueError(
                "Utonia cache preparation must run on a single GPU; use "
                "configs/accelerate/h200_single_gpu.yaml before distributed training"
            )
        prepare_cache(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
