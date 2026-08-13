"""Accelerate entry point for Wan point-cloud flow training."""

import argparse
import shutil
from pathlib import Path

from training.pc_config import load_pc_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def visualization_path(output_dir: str | Path, vis_dir: str, epoch: int) -> Path:
    return Path(output_dir) / vis_dir / f"epoch_{epoch:04d}.mp4"


def should_save_visualization(epoch: int, every_epochs: int) -> bool:
    """Return whether a completed one-based epoch is on the configured cadence."""
    if every_epochs <= 0:
        raise ValueError("every_epochs must be positive")
    return epoch % every_epochs == 0


def create_progress_bar(total: int, initial: int, enabled: bool):
    """Create a rank-zero progress bar over synchronized optimizer updates."""
    from tqdm.auto import tqdm

    return tqdm(
        total=total,
        initial=initial,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not enabled,
    )


def initialize_trackers(accelerator, config: dict) -> None:
    """Initialize the configured experiment tracker before logging metrics."""
    if config.get("report_to"):
        accelerator.init_trackers(config["tracker_project_name"], config=config)


def _pc_checkpoint_paths(output_dir: Path, setting: str | None) -> list[Path]:
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


def load_pc_checkpoint_with_fallback(
    accelerator, output_dir: Path, setting: str | None
) -> Path | None:
    """Load the requested PC state, falling back from incomplete latest checkpoints."""
    checkpoints = _pc_checkpoint_paths(output_dir, setting)
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
            print(f"Could not load {checkpoint}; trying the next most recent checkpoint: {error}")
        else:
            return checkpoint

    attempted = ", ".join(path.name for path, _ in failures)
    raise RuntimeError(
        f"Could not load any checkpoint selected by resume_from_checkpoint=latest: {attempted}"
    ) from failures[-1][1]


def prune_pc_checkpoints(root: str | Path, limit: int) -> None:
    """Keep only the newest numeric Accelerate PC checkpoint directories."""
    checkpoints = sorted(
        (
            path
            for path in Path(root).glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    for checkpoint in checkpoints[:-limit]:
        shutil.rmtree(checkpoint)


def create_pc_noise_scheduler(objective: dict):
    if objective["type"] == "flow":
        return None
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=objective["num_train_timesteps"],
        beta_schedule=objective["beta_schedule"],
        prediction_type="sample",
        clip_sample=False,
    )


def build_pc_training_dataset(
    config: dict,
    *,
    dataset_factory,
    extractor_factory,
    cache_preparer,
):
    """Create the baseline dataset or precompute the Utonia-backed variant."""
    data = config["data"]
    if not config["model"].get("utonia_enabled", False):
        return dataset_factory(data["dataset_root"]), None

    object_id = data["object_id"]
    cache_root = data["utonia_cache_root"]
    source_dataset = dataset_factory(data["dataset_root"], object_id=object_id)
    extractor = extractor_factory(cache_root)
    try:
        feature_dim = cache_preparer(source_dataset.source_paths, cache_root, extractor)
    finally:
        del extractor
    dataset = dataset_factory(
        data["dataset_root"],
        object_id=object_id,
        utonia_cache_root=cache_root,
    )
    return dataset, feature_dim


def main(config=None) -> None:
    if config is None:
        args = parse_args()
        config = load_pc_config(args.config, args.overrides)
    import torch
    import yaml
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from diffusers import DDIMScheduler
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    from training.pc_dataset import PCTrajectoryDataset
    from training.pc_ddpm import make_pc_ddpm_batch
    from training.pc_objectives import make_pc_flow_batch, mse_loss
    from training.schedules import create_lr_scheduler
    from training.pc_visualization import save_pointcloud_comparison_mp4
    from training.utonia_features import (
        UtoniaFeatureExtractor,
        prepare_utonia_feature_cache,
    )
    from wan.modules.pc_trajectory import PCTrajectoryModel
    from wan.pc_pipeline import PCDDIMPipeline, PCFlowPipeline
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with=config["report_to"] if config["report_to"] else None,
    )
    initialize_trackers(accelerator, config)
    set_seed(config["seed"])
    dataset, utonia_feature_dim = build_pc_training_dataset(
        config,
        dataset_factory=PCTrajectoryDataset,
        extractor_factory=UtoniaFeatureExtractor,
        cache_preparer=prepare_utonia_feature_cache,
    )
    loader = DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=config["dataloader_num_workers"])
    model_config = config["model"]
    objective = config["objective"]
    model = PCTrajectoryModel(n_points=config["data"]["num_points"], n_future_frames=48, latent_dim=model_config["latent_dim"], n_layers=model_config["n_layers"], num_heads=model_config["num_heads"], point_embed=model_config["point_embed"], objective_type=objective["type"], utonia_feature_dim=utonia_feature_dim)
    noise_scheduler = create_pc_noise_scheduler(objective)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(config["adam_beta1"], config["adam_beta2"]), weight_decay=config["adam_weight_decay"], eps=config["adam_epsilon"])
    scheduler = create_lr_scheduler(
        config["lr_scheduler"],
        optimizer,
        config["lr_warmup_steps"],
        config["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(config["seed"])
    resume_path = load_pc_checkpoint_with_fallback(
        accelerator, output_dir, config.get("resume_from_checkpoint")
    )
    step = int(resume_path.name.removeprefix("checkpoint-")) if resume_path else 0
    progress_bar = create_progress_bar(
        total=config["max_train_steps"],
        initial=step,
        enabled=accelerator.is_main_process,
    )
    for epoch in range(1, config["num_train_epochs"] + 1):
        visualization_batch = None
        for batch in loader:
            if visualization_batch is None:
                visualization_batch = {
                    key: value[:1].detach().cpu()
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
            with accelerator.accumulate(model):
                source = batch["points_src"].to(accelerator.device)
                if objective["type"] == "flow":
                    target_batch = make_pc_flow_batch(batch["points_tgt"].to(accelerator.device), source, generator, objective["time_shift"], objective["num_train_timesteps"])
                    target = target_batch.velocity_target
                else:
                    target_batch = make_pc_ddpm_batch(batch["points_tgt"].to(accelerator.device), noise_scheduler, generator)
                    target = target_batch.target
                utonia_features = batch.get("utonia_features")
                if utonia_features is not None:
                    utonia_features = utonia_features.to(accelerator.device)
                prediction = model(target_batch.model_input, target_batch.frame_times, source, batch["initial_linear_velocity"].to(accelerator.device), batch["initial_angular_velocity"].to(accelerator.device), utonia_features=utonia_features)
                loss = mse_loss(prediction, target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                accelerator.log({"train/loss": loss.detach().item(), "train/learning_rate": scheduler.get_last_lr()[0]}, step=step)
                if step % config["checkpointing_steps"] == 0:
                    accelerator.save_state(output_dir / f"checkpoint-{step}")
                    if accelerator.is_main_process:
                        prune_pc_checkpoints(
                            output_dir, config["checkpoints_total_limit"]
                        )
                if step >= config["max_train_steps"]:
                    break
        if (
            visualization_batch is not None
            and accelerator.is_main_process
            and should_save_visualization(
                epoch, config["visualization"]["every_epochs"]
            )
        ):
            unwrapped_model = accelerator.unwrap_model(model)
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            pipeline = (
                PCFlowPipeline(unwrapped_model, FlowUniPCMultistepScheduler(num_train_timesteps=objective["num_train_timesteps"], solver_order=config["sampling"]["solver_order"], prediction_type="flow_prediction", shift=1, use_dynamic_shifting=False), time_shift=objective["time_shift"])
                if objective["type"] == "flow"
                else PCDDIMPipeline(unwrapped_model, DDIMScheduler.from_config(noise_scheduler.config))
            )
            predicted_future = pipeline(
                visualization_batch["points_src"],
                visualization_batch["initial_linear_velocity"],
                visualization_batch["initial_angular_velocity"],
                accelerator.device,
                config["sampling"]["num_inference_steps"],
                torch.Generator(device=accelerator.device).manual_seed(config["seed"]),
                utonia_features=visualization_batch.get("utonia_features"),
            )
            predicted = torch.cat(
                (visualization_batch["points_src"].unsqueeze(1).to(accelerator.device), predicted_future), dim=1
            ).squeeze(0).cpu().numpy()
            ground_truth = torch.cat(
                (visualization_batch["points_src"].unsqueeze(1), visualization_batch["points_tgt"]), dim=1
            ).squeeze(0).numpy()
            save_pointcloud_comparison_mp4(
                predicted,
                ground_truth,
                visualization_path(output_dir, config["vis_dir"], epoch),
                config["visualization"]["fps"],
            )
            if was_training:
                unwrapped_model.train()
        if step >= config["max_train_steps"]:
            break
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
