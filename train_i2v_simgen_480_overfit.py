"""Overfit Wan2.2-TI2V-5B on native 480x480 SimGen sample_0 video frames."""

import argparse
import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.overfit_config import load_config
from training.schedules import create_lr_scheduler
from training.simgen_i2v_overfit_dataset import SimGenI2VOverfitDataset
from training.wan_i2v_training import (
    apply_classifier_free_dropout,
    classifier_free_guidance,
    expand_latent_timesteps,
    load_frozen_encoders,
    load_trainable_dit,
    make_flow_matching_batch,
    masked_velocity_mse,
    pin_history_latents,
)


HISTORY_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def create_optimizer(parameters, training: dict) -> torch.optim.AdamW:
    """Create AdamW using the configured I2V optimizer parameters."""
    return torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        betas=(training["adam_beta1"], training["adam_beta2"]),
        eps=training["adam_epsilon"],
        weight_decay=training["weight_decay"],
    )


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


def _checkpoint_paths(output_dir: Path, setting: str | None) -> list[Path]:
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


def load_checkpoint_with_fallback(accelerator, output_dir: Path, setting: str | None) -> Path | None:
    """Load requested state, skipping incomplete checkpoints for latest."""
    checkpoints = _checkpoint_paths(output_dir, setting)
    if not checkpoints:
        return None

    errors = []
    for checkpoint in checkpoints:
        try:
            accelerator.load_state(checkpoint)
        except Exception as error:
            if setting != "latest":
                raise
            errors.append((checkpoint, error))
            print(f"Could not load {checkpoint}; trying prior checkpoint: {error}")
        else:
            return checkpoint

    attempted = ", ".join(path.name for path, _ in errors)
    raise RuntimeError(f"Could not load any latest checkpoint: {attempted}") from errors[-1][1]


def _encode_batch(vae, videos: torch.Tensor) -> torch.Tensor:
    clips = [clip.permute(1, 0, 2, 3).contiguous() for clip in videos]
    with torch.no_grad():
        return torch.stack(vae.encode(clips))


def _token_timesteps(latent_timesteps: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return expand_latent_timesteps(
        latent_timesteps,
        latent_height=latents.shape[-2],
        latent_width=latents.shape[-1],
    )


def make_history_conditioned_visualization_latent(
    clean_latents: torch.Tensor,
    generator: torch.Generator,
    history_frames: int = HISTORY_FRAMES,
) -> torch.Tensor:
    """Create seeded latent noise with the requested clean history prefix."""
    if clean_latents.ndim != 4:
        raise ValueError("clean_latents must have shape [C, T, H, W]")
    noisy_latent = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    return pin_history_latents(noisy_latent, clean_latents, history_frames)


@torch.no_grad()
def sample_visualization_latent(
    model,
    vae,
    text_encoder,
    video: torch.Tensor,
    prompt: str,
    unconditional_prompt: str,
    wan_config,
    time_shift: float,
    seed: int,
    cfg_scale: float,
    history_frames: int = HISTORY_FRAMES,
) -> torch.Tensor:
    """Generate a fixed-seed sample while pinning four clean latent slots."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    was_training = model.training
    model.eval()
    device = video.device
    generator = torch.Generator(device=device).manual_seed(seed)
    clean_latents = vae.encode([video.permute(1, 0, 2, 3).contiguous()])[0]
    latent = make_history_conditioned_visualization_latent(
        clean_latents, generator, history_frames
    )
    conditional_context = text_encoder([prompt], device)
    unconditional_context = text_encoder([unconditional_prompt], device)
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=wan_config.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(50, device=device, shift=time_shift)
    seq_len = latent.shape[1] * (latent.shape[-2] // 2) * (latent.shape[-1] // 2)
    with torch.autocast(device_type=device.type, dtype=wan_config.param_dtype):
        for timestep in scheduler.timesteps:
            frame_times = torch.full(
                (1, latent.shape[1]),
                timestep.item(),
                device=device,
                dtype=latent.dtype,
            )
            frame_times[:, :history_frames] = 0
            conditional_velocity = model(
                [latent],
                t=_token_timesteps(frame_times, latent.unsqueeze(0)),
                context=conditional_context,
                seq_len=seq_len,
            )[0]
            unconditional_velocity = model(
                [latent],
                t=_token_timesteps(frame_times, latent.unsqueeze(0)),
                context=unconditional_context,
                seq_len=seq_len,
            )[0]
            prediction = classifier_free_guidance(
                unconditional_velocity, conditional_velocity, cfg_scale
            )
            latent = scheduler.step(
                prediction.unsqueeze(0),
                timestep,
                latent.unsqueeze(0),
                return_dict=False,
                generator=generator,
            )[0].squeeze(0)
            latent = pin_history_latents(latent, clean_latents, history_frames)
    if was_training:
        model.train()
    return latent


def save_visualization(vae, latent: torch.Tensor, output_file: Path, fps: int) -> None:
    """Decode a latent and write an MP4 reference or generated visualization."""
    from imageio.v2 import get_writer

    video = vae.decode([latent])[0].permute(1, 2, 3, 0)
    frames = ((video.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with get_writer(output_file, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def visualization_path(output_dir: str | Path, step: int) -> Path:
    """Return the fixed-step qualitative-video path."""
    return Path(output_dir) / "vis" / f"step_{step:05d}.mp4"


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )
    from wan.configs.wan_ti2v_5B import ti2v_5B

    config = load_config(args.config, args.overrides)
    training, data, logging = config["training"], config["data"], config["logging"]
    unconditional_prompt = (
        ti2v_5B.sample_neg_prompt
        if training.get("use_wan_negative_prompt", False)
        else ""
    )
    output_dir = Path(logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"],
        log_with="wandb",
    )
    set_seed(training["seed"])
    init_kwargs = {"wandb": {}}
    if logging.get("wandb_run_name"):
        init_kwargs["wandb"]["name"] = logging["wandb_run_name"]
    accelerator.init_trackers(
        logging["wandb_project"], config=config, init_kwargs=init_kwargs
    )

    dataset = SimGenI2VOverfitDataset(data["sample_root"], data["prompt"])
    dataloader = DataLoader(
        dataset,
        batch_size=training["train_batch_size"],
        shuffle=False,
        num_workers=data["dataloader_num_workers"],
        pin_memory=True,
    )
    vae, text_encoder = load_frozen_encoders(
        config["model"]["checkpoint_dir"], ti2v_5B, accelerator.device
    )
    model = load_trainable_dit(
        config["model"]["checkpoint_dir"], config["model"]["gradient_checkpointing"]
    )
    optimizer = create_optimizer(model.parameters(), training)
    scheduler = create_lr_scheduler(
        training["lr_scheduler"],
        optimizer,
        training["warmup_steps"],
        training["max_train_steps"],
        cosine_factory=get_cosine_schedule_with_warmup,
        constant_factory=get_constant_schedule_with_warmup,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(training["seed"])
    global_step = 0
    resume_path = load_checkpoint_with_fallback(
        accelerator, output_dir, training.get("resume_from_checkpoint")
    )
    if resume_path:
        global_step = int(resume_path.name.removeprefix("checkpoint-"))
    steps_per_epoch = math.ceil(
        len(dataloader) / training["gradient_accumulation_steps"]
    )
    progress_bar = create_progress_bar(
        total=training["max_train_steps"],
        initial=global_step,
        enabled=accelerator.is_main_process,
    )
    target_saved = False

    while global_step < training["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(model):
                videos = batch["video"].to(accelerator.device, non_blocking=True)
                clean_latents = _encode_batch(vae, videos)
                if accelerator.is_main_process and not target_saved:
                    save_visualization(
                        vae,
                        clean_latents[0],
                        output_dir / "vis" / "target.mp4",
                        ti2v_5B.sample_fps,
                    )
                    target_saved = True
                with torch.no_grad():
                    context = text_encoder(list(batch["prompt"]), accelerator.device)
                    drop_mask = torch.rand(
                        len(context),
                        device=accelerator.device,
                        generator=generator,
                    ) < training["text_dropout_probability"]
                    if drop_mask.any():
                        unconditional_context = text_encoder(
                            [unconditional_prompt] * len(context), accelerator.device
                        )
                        context = apply_classifier_free_dropout(
                            context, unconditional_context, drop_mask
                        )
                flow = make_flow_matching_batch(
                    clean_latents,
                    generator,
                    training["time_shift"],
                    training["num_train_timesteps"],
                    history_frames=HISTORY_FRAMES,
                )
                token_times = _token_timesteps(flow.latent_timesteps, clean_latents)
                with accelerator.autocast():
                    prediction = torch.stack(
                        model(
                            [
                                flow.model_input[index]
                                for index in range(videos.shape[0])
                            ],
                            t=token_times,
                            context=context,
                            seq_len=token_times.shape[1],
                        )
                    )
                    loss = masked_velocity_mse(
                        prediction, flow.velocity_target, flow.loss_mask
                    )
                accelerator.backward(loss)
                gradient_norm = None
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(
                        model.parameters(), training["max_grad_norm"]
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            accelerator.log(
                {
                    "train/loss": loss.detach().item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gradient_norm": gradient_norm.detach().item(),
                },
                step=global_step,
            )
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if global_step % training["checkpoint_every_steps"] == 0:
                accelerator.save_state(output_dir / f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_checkpoints(
                        output_dir, training["checkpoints_total_limit"]
                    )
            if (
                accelerator.is_main_process
                and global_step % training["visualization_every_steps"] == 0
            ):
                latent = sample_visualization_latent(
                    accelerator.unwrap_model(model),
                    vae,
                    text_encoder,
                    videos[0],
                    data["prompt"],
                    unconditional_prompt,
                    ti2v_5B,
                    training["time_shift"],
                    training["visualization_seed"],
                    training["visualization_cfg_scale"],
                )
                save_visualization(
                    vae,
                    latent,
                    visualization_path(output_dir, global_step),
                    ti2v_5B.sample_fps,
                )
            if global_step >= training["max_train_steps"]:
                break

    accelerator.wait_for_everyone()
    progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()

