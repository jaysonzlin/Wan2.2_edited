"""Accelerate entrypoint for overfitting Wan2.2-TI2V-5B on Kubric sequences."""

import argparse
import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.overfit_config import load_config
from training.overfit_dataset import KubricI2VOverfitDataset
from training.wan_i2v_training import (
    expand_latent_timesteps,
    load_frozen_encoders,
    load_trainable_dit,
    make_flow_matching_batch,
    masked_velocity_mse,
)


def visualization_path(output_dir: str | Path, epoch: int) -> Path:
    """Return the required local-only qualitative-video path."""
    return Path(output_dir) / "vis" / f"epoch_{epoch:04d}.mp4"


def create_progress_bar(total: int, initial: int, enabled: bool):
    """Create a rank-zero optimizer-step progress bar."""
    from tqdm.auto import tqdm

    return tqdm(
        total=total,
        initial=initial,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not enabled,
    )


def prune_checkpoints(root: str | Path, limit: int) -> None:
    """Keep only the newest numeric Accelerate checkpoint directories."""
    checkpoints = sorted(
        (path for path in Path(root).glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    for checkpoint in checkpoints[:-limit]:
        shutil.rmtree(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _encode_batch(vae, videos: torch.Tensor) -> torch.Tensor:
    clips = [clip.permute(1, 0, 2, 3).contiguous() for clip in videos]
    with torch.no_grad():
        return torch.stack(vae.encode(clips))


def _token_timesteps(latent_timesteps: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return expand_latent_timesteps(
        latent_timesteps, latent_height=latents.shape[-2], latent_width=latents.shape[-1]
    )


@torch.no_grad()
def save_visualization(
    model, vae, text_encoder, condition_frame, prompt, output_file, wan_config,
    time_shift, num_frames, seed,
) -> None:
    """Generate one deterministic, local-only native TI2V sample."""
    from imageio.v2 import get_writer
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    was_training = model.training
    model.eval()
    device = condition_frame.device
    generator = torch.Generator(device=device).manual_seed(seed)
    condition_latent = vae.encode([condition_frame.unsqueeze(1)])[0]
    latent = torch.randn(
        (condition_latent.shape[0], (num_frames - 1) // 4 + 1,
         condition_latent.shape[-2], condition_latent.shape[-1]),
        device=device, dtype=condition_latent.dtype, generator=generator,
    )
    latent[:, :1] = condition_latent
    context = text_encoder([prompt], device)
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=wan_config.num_train_timesteps, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(50, device=device, shift=time_shift)
    seq_len = latent.shape[1] * (latent.shape[-2] // 2) * (latent.shape[-1] // 2)
    with torch.autocast(device_type=device.type, dtype=wan_config.param_dtype):
        for timestep in scheduler.timesteps:
            frame_times = torch.full(
                (1, latent.shape[1]), timestep.item(), device=device, dtype=latent.dtype
            )
            frame_times[:, 0] = 0
            prediction = model(
                [latent], t=_token_timesteps(frame_times, latent.unsqueeze(0)),
                context=context, seq_len=seq_len,
            )[0]
            latent = scheduler.step(
                prediction.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False, generator=generator
            )[0].squeeze(0)
            latent[:, :1] = condition_latent
    video = vae.decode([latent])[0].permute(1, 2, 3, 0)
    frames = ((video.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with get_writer(output_file, fps=wan_config.sample_fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    if was_training:
        model.train()


def _checkpoint_path(output_dir: Path, setting: str | None) -> Path | None:
    if not setting:
        return None
    if setting != "latest":
        return Path(setting)
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.removeprefix("checkpoint-"))
    )
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup
    from wan.configs.wan_ti2v_5B import ti2v_5B

    config = load_config(args.config, args.overrides)
    training, data, logging = config["training"], config["data"], config["logging"]
    output_dir = Path(logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"], log_with="wandb",
    )
    set_seed(training["seed"])
    init_kwargs = {"wandb": {}}
    if logging.get("wandb_run_name"):
        init_kwargs["wandb"]["name"] = logging["wandb_run_name"]
    accelerator.init_trackers(logging["wandb_project"], config=config, init_kwargs=init_kwargs)

    dataset = KubricI2VOverfitDataset(data["dataset_root"], data["prompt"])
    dataloader = DataLoader(
        dataset, batch_size=training["train_batch_size"], shuffle=True,
        num_workers=data["dataloader_num_workers"], pin_memory=True,
    )
    vae, text_encoder = load_frozen_encoders(config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device)
    model = load_trainable_dit(config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, training["warmup_steps"], training["max_train_steps"])
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    generator = torch.Generator(device=accelerator.device).manual_seed(training["seed"])
    global_step = 0
    resume_path = _checkpoint_path(output_dir, training.get("resume_from_checkpoint"))
    if resume_path:
        accelerator.load_state(resume_path)
        global_step = int(resume_path.name.removeprefix("checkpoint-"))
    steps_per_epoch = math.ceil(len(dataloader) / training["gradient_accumulation_steps"])
    progress_bar = create_progress_bar(
        total=training["max_train_steps"],
        initial=global_step,
        enabled=accelerator.is_main_process,
    )

    while global_step < training["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(model):
                videos = batch["video"].to(accelerator.device, non_blocking=True)
                clean_latents = _encode_batch(vae, videos)
                with torch.no_grad():
                    context = text_encoder(list(batch["prompt"]), accelerator.device)
                flow = make_flow_matching_batch(clean_latents, generator, training["time_shift"], training["num_train_timesteps"])
                token_times = _token_timesteps(flow.latent_timesteps, clean_latents)
                with accelerator.autocast():
                    prediction = torch.stack(model(
                        [flow.model_input[index] for index in range(videos.shape[0])],
                        t=token_times, context=context, seq_len=token_times.shape[1],
                    ))
                    loss = masked_velocity_mse(prediction, flow.velocity_target, flow.loss_mask)
                accelerator.backward(loss)
                gradient_norm = None
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            accelerator.log({
                "train/loss": loss.detach().item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/gradient_norm": gradient_norm.detach().item(),
            }, step=global_step)
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            epoch = math.ceil(global_step / steps_per_epoch)
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_checkpoints(output_dir, training["checkpoints_total_limit"])
            if accelerator.is_main_process and global_step % training["visualization_every_steps"] == 0:
                save_visualization(
                    accelerator.unwrap_model(model), vae, text_encoder, videos[0, 0], data["prompt"],
                    visualization_path(output_dir, epoch), ti2v_5B, training["time_shift"], data["num_frames"], training["seed"],
                )
            if global_step >= training["max_train_steps"]:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_dir / "final_dit", safe_serialization=True)
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
