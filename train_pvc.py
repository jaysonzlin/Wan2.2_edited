"""Accelerate entry point for dedicated point-view-conditioned PC DDPM training."""

import argparse
from pathlib import Path

from training.pvc_config import load_pvc_config
from training.schedules import create_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_pvc_training_dataset(config, *, dataset_factory, extractor_factory, trajectory_cache_preparer, point_view_cache_preparer):
    data = config["data"]
    source = dataset_factory(data["dataset_root"], object_id=data["object_id"])
    extractor = extractor_factory(data["utonia_cache_root"], config["seed"])
    try:
        feature_dim = trajectory_cache_preparer(source.source_paths, data["utonia_cache_root"], extractor)
        point_view_cache_preparer(
            source.point_view_source_paths, data["point_view_utonia_cache_root"], extractor,
            feature_dim=feature_dim,
        )
    finally:
        del extractor
    return dataset_factory(
        data["dataset_root"], object_id=data["object_id"],
        utonia_cache_root=data["utonia_cache_root"],
        point_view_utonia_cache_root=data["point_view_utonia_cache_root"],
        point_view_feature_dim=feature_dim,
    ), feature_dim


def compute_pvc_training_prediction(batch, model, noise_scheduler, generator, device, *, ddpm_batch_factory):
    future = batch["points_tgt"].to(device)
    objective_batch = ddpm_batch_factory(future, noise_scheduler, generator, known_frames=4)
    prediction = model(
        objective_batch.model_input, objective_batch.frame_times, batch["points_history"].to(device),
        batch["point_views"].to(device), batch["point_view_mask"].to(device),
        batch["utonia_features"].to(device), batch["point_view_utonia_features"].to(device),
    )
    return prediction, objective_batch.target


def build_pvc_lr_scheduler(schedule_name, optimizer, warmup_steps, max_train_steps, *, cosine_factory, constant_factory):
    """Create PVC's scheduler using the same factory contract as train_pc."""
    return create_lr_scheduler(
        schedule_name, optimizer, warmup_steps, max_train_steps,
        cosine_factory=cosine_factory, constant_factory=constant_factory,
    )


def sample_pvc_visualization(pipeline, batch, device, num_inference_steps, generator):
    """Sample PVC futures and assemble the complete 49-frame trajectory."""
    import torch

    history = batch["points_history"]
    predicted_future = pipeline(
        history, batch["point_views"], batch["point_view_mask"],
        batch["utonia_features"], batch["point_view_utonia_features"],
        device, num_inference_steps, generator,
    )
    predicted = torch.cat((history.to(device), predicted_future), dim=1).squeeze(0).cpu()
    ground_truth = torch.cat((history, batch["points_tgt"]), dim=1).squeeze(0).cpu()
    return predicted, ground_truth


def main(config=None):
    if config is None:
        args = parse_args()
        config = load_pvc_config(args.config, args.overrides)
    import torch
    import yaml
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from diffusers import DDIMScheduler
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    from training.pc_ddpm import make_pc_ddpm_batch
    from training.pc_objectives import mse_loss
    from training.pvc_dataset import PVCTrajectoryDataset
    from training.pvc_utonia_features import prepare_point_view_utonia_feature_cache
    from training.utonia_features import UtoniaFeatureExtractor, prepare_utonia_feature_cache
    from wan.modules.pvc_trajectory import PVCTrajectoryModel
    from wan.pvc_pipeline import PVCHistoryDDIMPipeline
    from training.pc_visualization import save_pointcloud_comparison_mp4
    from train_pc import (
        create_pc_noise_scheduler, create_progress_bar, first_unfinished_epoch,
        load_pc_checkpoint_with_fallback, prune_pc_checkpoints,
        should_save_visualization, visualization_path,
    )

    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as handle: yaml.safe_dump(config, handle)
    accelerator = Accelerator(gradient_accumulation_steps=config["gradient_accumulation_steps"], mixed_precision=config["mixed_precision"], log_with=config["report_to"] or None)
    if config.get("report_to"): accelerator.init_trackers(config["tracker_project_name"], config=config)
    set_seed(config["seed"])
    dataset, feature_dim = build_pvc_training_dataset(config, dataset_factory=PVCTrajectoryDataset, extractor_factory=UtoniaFeatureExtractor, trajectory_cache_preparer=prepare_utonia_feature_cache, point_view_cache_preparer=prepare_point_view_utonia_feature_cache)
    loader = DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=config["dataloader_num_workers"])
    model = PVCTrajectoryModel(n_points=2048, n_future_frames=45, latent_dim=256, n_layers=8, num_heads=4, utonia_feature_dim=feature_dim)
    scheduler = create_pc_noise_scheduler(config["objective"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(config["adam_beta1"], config["adam_beta2"]), weight_decay=config["adam_weight_decay"], eps=config["adam_epsilon"])
    lr_scheduler = build_pvc_lr_scheduler(
        config["lr_scheduler"], optimizer, config["lr_warmup_steps"], config["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    model, optimizer, loader, lr_scheduler = accelerator.prepare(model, optimizer, loader, lr_scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(config["seed"])
    resumed = load_pc_checkpoint_with_fallback(accelerator, output_dir, config.get("resume_from_checkpoint")); step = int(resumed.name.removeprefix("checkpoint-")) if resumed else 0
    progress_bar = create_progress_bar(config["max_train_steps"], step, accelerator.is_main_process)
    for epoch in range(first_unfinished_epoch(step), config["num_train_epochs"] + 1):
        visualization_batch = None
        for batch in loader:
            if visualization_batch is None:
                visualization_batch = {key: value[:1].detach().cpu() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            with accelerator.accumulate(model):
                prediction, target = compute_pvc_training_prediction(batch, model, scheduler, generator, accelerator.device, ddpm_batch_factory=make_pc_ddpm_batch)
                accelerator.backward(mse_loss(prediction, target))
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                loss = mse_loss(prediction, target)
                progress_bar.update(1); progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                accelerator.log({"train/loss": loss.detach().item(), "train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=step)
                if step % config["checkpointing_steps"] == 0:
                    accelerator.save_state(output_dir / f"checkpoint-{step}")
                    if accelerator.is_main_process: prune_pc_checkpoints(output_dir, config["checkpoints_total_limit"])
                if step >= config["max_train_steps"]: break
        if visualization_batch is not None and accelerator.is_main_process and should_save_visualization(epoch, config["visualization"]["every_epochs"]):
            unwrapped = accelerator.unwrap_model(model); was_training = unwrapped.training; unwrapped.eval()
            pipeline = PVCHistoryDDIMPipeline(unwrapped, DDIMScheduler.from_config(scheduler.config))
            predicted, ground_truth = sample_pvc_visualization(pipeline, visualization_batch, accelerator.device, config["sampling"]["num_inference_steps"], torch.Generator(device=accelerator.device).manual_seed(config["seed"]))
            save_pointcloud_comparison_mp4(predicted.numpy(), ground_truth.numpy(), visualization_path(output_dir, config["vis_dir"], epoch), config["visualization"]["fps"])
            if was_training: unwrapped.train()
        if step >= config["max_train_steps"]: break
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
