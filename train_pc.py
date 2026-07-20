"""Accelerate entry point for Wan point-cloud flow training."""

import argparse
from pathlib import Path

from training.pc_config import load_pc_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def visualization_path(output_dir: str | Path, vis_dir: str, epoch: int) -> Path:
    return Path(output_dir) / vis_dir / f"epoch_{epoch:04d}.mp4"


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


def main(config=None) -> None:
    if config is None:
        args = parse_args()
        config = load_pc_config(args.config, args.overrides)
    import torch
    import yaml
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from training.pc_dataset import PCTrajectoryDataset
    from training.pc_flow import flow_mse, make_pc_flow_batch
    from wan.modules.pc_flow import PCFlowModel

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with=config["report_to"] if config["report_to"] else None,
    )
    set_seed(config["seed"])
    dataset = PCTrajectoryDataset(config["data"]["dataset_root"])
    loader = DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=config["dataloader_num_workers"])
    model_config = config["model"]
    model = PCFlowModel(n_points=config["data"]["num_points"], n_future_frames=48, latent_dim=model_config["latent_dim"], n_layers=model_config["n_layers"], num_heads=model_config["num_heads"], point_embed=model_config["point_embed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(config["adam_beta1"], config["adam_beta2"]), weight_decay=config["adam_weight_decay"], eps=config["adam_epsilon"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, config["lr_warmup_steps"], config["max_train_steps"])
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(config["seed"])
    step = 0
    progress_bar = create_progress_bar(
        total=config["max_train_steps"],
        initial=step,
        enabled=accelerator.is_main_process,
    )
    while step < config["max_train_steps"]:
        for batch in loader:
            with accelerator.accumulate(model):
                source = batch["points_src"].to(accelerator.device)
                flow = make_pc_flow_batch(batch["points_tgt"].to(accelerator.device), source, generator, config["flow"]["time_shift"], config["flow"]["num_train_timesteps"])
                prediction = model(flow.model_input, flow.frame_times, source, batch["initial_linear_velocity"].to(accelerator.device), batch["initial_angular_velocity"].to(accelerator.device))
                loss = flow_mse(prediction, flow.velocity_target)
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
                if step >= config["max_train_steps"]:
                    break
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
