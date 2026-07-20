"""Sample a fine-tuned TI2V checkpoint across shift and step-count experiments."""

import argparse
from pathlib import Path


BASE_DIR = Path("Wan2.2-TI2V-5B")
MODEL_DIR = Path("outputs/checkpoint-8750")
OUT_DIR = Path("outputs/inference_sweep")

PROMPT = "Objects moving in a Kubric simulator"
SEED = 42
NUM_FRAMES = 49
CFG_SCALE = 1.0

# The first five entries vary shift at a fixed 50 denoising steps.  The last
# three vary steps at the fixed native shift of 5, avoiding a duplicate 5/50 run.
EXPERIMENTS = (
    (1, 50),
    (2, 50),
    (3, 50),
    (4, 50),
    (5, 50),
    (5, 100),
    (5, 150),
    (5, 200),
)


def output_name(shift: int, num_steps: int) -> str:
    """Return the per-sample MP4 filename for one inference experiment."""
    return f"shift_{shift}_steps_{num_steps}.mp4"


def configure_scheduler(scheduler, num_steps: int, device, shift: int) -> None:
    """Configure sampling from the first sigma, including duplicate timesteps."""
    scheduler.set_timesteps(num_steps, device=device, shift=shift)
    scheduler.set_begin_index(0)


def parse_args() -> argparse.Namespace:
    """Parse lightweight utility options without importing CUDA dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Print output filenames and exit without loading model dependencies.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the selected checkpoint and write one MP4 for every experiment."""
    args = parse_args()
    if args.list_experiments:
        for shift, num_steps in EXPERIMENTS:
            print(output_name(shift, num_steps))
        return

    import imageio.v2 as imageio
    import torch
    from safetensors.torch import load_file

    from train_i2v import _token_timesteps
    from training.overfit_dataset import KubricI2VOverfitDataset
    from training.wan_i2v_training import (
        classifier_free_guidance,
        load_frozen_encoders,
        load_trainable_dit,
    )
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = torch.device("cuda")
    weights_path = MODEL_DIR / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights: {weights_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = KubricI2VOverfitDataset("training_dataset", PROMPT)
    condition_frame = dataset[0]["video"][0].to(device)
    vae, text_encoder = load_frozen_encoders(BASE_DIR, ti2v_5B, device)

    model = load_trainable_dit(BASE_DIR, gradient_checkpointing=False).to(device)
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    model.eval()

    with torch.inference_mode():
        condition_latent = vae.encode([condition_frame.unsqueeze(1)])[0]
        conditional_context = text_encoder([PROMPT], device)
        unconditional_context = text_encoder([""], device)

        for shift, num_steps in EXPERIMENTS:
            generator = torch.Generator(device=device).manual_seed(SEED)
            latent = torch.randn(
                (
                    condition_latent.shape[0],
                    (NUM_FRAMES - 1) // 4 + 1,
                    condition_latent.shape[-2],
                    condition_latent.shape[-1],
                ),
                device=device,
                dtype=condition_latent.dtype,
                generator=generator,
            )
            latent[:, :1] = condition_latent

            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=ti2v_5B.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            configure_scheduler(scheduler, num_steps, device, shift)
            seq_len = (
                latent.shape[1]
                * (latent.shape[-2] // 2)
                * (latent.shape[-1] // 2)
            )

            with torch.autocast(device_type="cuda", dtype=ti2v_5B.param_dtype):
                for timestep in scheduler.timesteps:
                    frame_times = torch.full(
                        (1, latent.shape[1]),
                        timestep.item(),
                        device=device,
                        dtype=latent.dtype,
                    )
                    frame_times[:, 0] = 0
                    conditional = model(
                        [latent],
                        t=_token_timesteps(frame_times, latent.unsqueeze(0)),
                        context=conditional_context,
                        seq_len=seq_len,
                    )[0]
                    unconditional = model(
                        [latent],
                        t=_token_timesteps(frame_times, latent.unsqueeze(0)),
                        context=unconditional_context,
                        seq_len=seq_len,
                    )[0]
                    prediction = classifier_free_guidance(
                        unconditional, conditional, CFG_SCALE
                    )
                    latent = scheduler.step(
                        prediction.unsqueeze(0),
                        timestep,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0].squeeze(0)
                    latent[:, :1] = condition_latent

            video = vae.decode([latent])[0].permute(1, 2, 3, 0)
            frames = ((video.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
            output_path = OUT_DIR / output_name(shift, num_steps)
            with imageio.get_writer(
                output_path,
                fps=ti2v_5B.sample_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
