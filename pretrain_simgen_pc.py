"""Pretrain the joint SimGen point-cloud trajectory model without Wan."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_pc import create_progress_bar, load_pc_checkpoint_with_fallback, prune_pc_checkpoints
from training.simgen_joint_dataset import SimGenJointDataset, simgen_joint_collate
from training.simgen_pc_pretraining import (
    flatten_simgen_pc_batch,
    load_best_export_metadata,
    object_mse_totals,
    pc_pretraining_prediction,
    reduce_object_mean,
    save_best_pc_export,
    save_sample_zero_visualizations,
)
from training.simgen_pc_pretraining_config import load_simgen_pc_pretraining_config
from training.simgen_utonia_features import SimGenUtoniaCache


def build_datasets(config: dict) -> tuple[SimGenJointDataset, SimGenJointDataset]:
    """Build the fixed SimGen PC pretraining and validation splits."""
    data = config["data"]
    common = {
        "expected_points": data["num_points"],
        "utonia_cache_root": data["utonia_cache_root"],
    }
    return (
        SimGenJointDataset(
            data["dataset_root"], list(range(data["train_start"], data["train_end"] + 1)), **common
        ),
        SimGenJointDataset(
            data["dataset_root"],
            list(range(data["validation_start"], data["validation_end"] + 1)),
            **common,
        ),
    )


def _make_model(feature_width: int, config: dict):
    from wan.modules.pc_trajectory import PCTrajectoryModel

    model_config = config["model"]
    data = config["data"]
    return PCTrajectoryModel(
        n_points=data["num_points"],
        n_future_frames=45,
        latent_dim=model_config["latent_dim"],
        n_layers=model_config["n_layers"],
        num_heads=model_config["num_heads"],
        objective_type="ddpm",
        conditioning="history",
        history_frames=model_config["history_frames"],
        utonia_feature_dim=feature_width,
    )


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def _validate(model, validation_loader, scheduler, generator, device, accelerator) -> torch.Tensor:
    local_sum = torch.zeros((), device=device)
    local_count = 0
    for raw_batch in validation_loader:
        batch = flatten_simgen_pc_batch(raw_batch, device)
        prediction, target = pc_pretraining_prediction(batch, model, scheduler, generator, device)
        batch_sum, batch_count = object_mse_totals(prediction, target)
        local_sum += batch_sum
        local_count += batch_count
    return reduce_object_mean(accelerator, local_sum, local_count)


def run_training(config: dict) -> None:
    """Run the fixed distributed SimGen PC pretraining experiment."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDIMScheduler, DDPMScheduler
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    import yaml

    from training.schedules import create_lr_scheduler
    from wan.pc_pipeline import PCHistoryDDIMPipeline

    data, training, logging = config["data"], config["training"], config["logging"]
    SimGenUtoniaCache(data["utonia_cache_root"])
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

    loader_kwargs = {
        "batch_size": data["train_batch_size"],
        "num_workers": data["dataloader_num_workers"],
        "pin_memory": True,
        "collate_fn": simgen_joint_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = _make_model(feature_width, config)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    lr_scheduler = create_lr_scheduler(
        training["lr_scheduler"],
        optimizer,
        training["warmup_steps"],
        training["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["objective"]["num_train_timesteps"],
        beta_schedule=config["objective"]["beta_schedule"],
        prediction_type="sample",
        clip_sample=False,
    )
    model, optimizer, train_loader, validation_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, lr_scheduler
    )
    resumed = load_pc_checkpoint_with_fallback(
        accelerator, output_dir, training.get("resume_from_checkpoint")
    )
    global_step = int(resumed.name.removeprefix("checkpoint-")) if resumed else 0
    best_loss, _best_step = load_best_export_metadata(output_dir)
    generator = _make_generator(accelerator.device, training["seed"])
    validation_generator = _make_generator(accelerator.device, training["seed"])
    progress_bar = create_progress_bar(training["max_train_steps"], global_step, accelerator.is_main_process)

    while global_step < training["max_train_steps"]:
        for raw_batch in train_loader:
            with accelerator.accumulate(model):
                batch = flatten_simgen_pc_batch(raw_batch, accelerator.device)
                prediction, target = pc_pretraining_prediction(
                    batch, model, noise_scheduler, generator, accelerator.device
                )
                loss_sum, object_count = object_mse_totals(prediction, target)
                loss = loss_sum / object_count
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            reduced_loss = reduce_object_mean(accelerator, loss_sum.detach(), object_count)
            if accelerator.is_main_process:
                metrics = {
                    "train/pc_loss": reduced_loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(metrics, step=global_step)
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{metrics['train/pc_loss']:.4f}")
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_pc_checkpoints(output_dir, training["checkpoints_total_limit"])
                accelerator.wait_for_everyone()
            if global_step % config["validation"]["every_steps"] == 0:
                unwrapped = accelerator.unwrap_model(model)
                was_training = unwrapped.training
                unwrapped.eval()
                validation_loss = _validate(
                    unwrapped, validation_loader, noise_scheduler, validation_generator,
                    accelerator.device, accelerator,
                )
                if accelerator.is_main_process:
                    accelerator.log({"validation/pc_loss": validation_loss.item()}, step=global_step)
                    if save_best_pc_export(
                        unwrapped,
                        output_dir,
                        validation_loss=validation_loss.item(),
                        step=global_step,
                        best_loss=best_loss,
                    ):
                        best_loss = validation_loss.item()
                    visualization = flatten_simgen_pc_batch(visualization_batch, accelerator.device)
                    save_sample_zero_visualizations(
                        PCHistoryDDIMPipeline(
                            unwrapped,
                            DDIMScheduler.from_config(noise_scheduler.config),
                        ),
                        visualization,
                        output_dir,
                        step=global_step,
                        fps=config["visualization"]["fps"],
                        device=accelerator.device,
                        num_inference_steps=config["sampling"]["num_inference_steps"],
                        generator=_make_generator(accelerator.device, training["seed"]),
                    )
                accelerator.wait_for_everyone()
                if was_training:
                    unwrapped.train()
            if global_step >= training["max_train_steps"]:
                break
    progress_bar.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_training(load_simgen_pc_pretraining_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
