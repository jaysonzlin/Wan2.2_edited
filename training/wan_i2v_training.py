"""Wan TI2V-specific model loading and flow-matching training helpers."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class FlowMatchingBatch:
    """Noised latents and the masked velocity target for one training update."""

    model_input: torch.Tensor
    velocity_target: torch.Tensor
    latent_timesteps: torch.Tensor
    loss_mask: torch.Tensor


def make_flow_matching_batch(
    clean_latents: torch.Tensor,
    generator: torch.Generator,
    time_shift: float,
    num_train_timesteps: int,
) -> FlowMatchingBatch:
    """Create a Wan flow-matching objective with a clean conditioned first slot."""
    if clean_latents.ndim != 5:
        raise ValueError("clean_latents must have shape [batch, channels, time, height, width]")
    if clean_latents.shape[2] < 2:
        raise ValueError("I2V training requires a conditioned slot and target slots")

    batch_size = clean_latents.shape[0]
    uniform_times = torch.rand(
        batch_size, device=clean_latents.device, generator=generator, dtype=clean_latents.dtype
    )
    shifted_times = time_shift * uniform_times / (1 + (time_shift - 1) * uniform_times)
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    interpolation = shifted_times.view(batch_size, 1, 1, 1, 1)
    model_input = (1 - interpolation) * clean_latents + interpolation * noise
    velocity_target = noise - clean_latents

    loss_mask = torch.ones_like(clean_latents, dtype=torch.bool)
    loss_mask[:, :, 0] = False
    model_input[:, :, 0] = clean_latents[:, :, 0]

    latent_timesteps = shifted_times.mul(num_train_timesteps).unsqueeze(1).expand(
        -1, clean_latents.shape[2]
    )
    latent_timesteps[:, 0] = 0
    return FlowMatchingBatch(
        model_input=model_input,
        velocity_target=velocity_target,
        latent_timesteps=latent_timesteps,
        loss_mask=loss_mask,
    )


def masked_velocity_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return mean squared velocity error over the noised latent elements only."""
    float_mask = mask.to(dtype=prediction.dtype)
    return ((prediction - target).pow(2) * float_mask).sum() / float_mask.sum().clamp_min(1)


def expand_latent_timesteps(
    latent_timesteps: torch.Tensor, latent_height: int, latent_width: int, patch_size: int = 2
) -> torch.Tensor:
    """Expand per-latent-frame time values to the Wan DiT token sequence."""
    if latent_height % patch_size or latent_width % patch_size:
        raise ValueError("Latent spatial dimensions must be divisible by the Wan patch size")
    return (
        latent_timesteps[:, :, None, None]
        .expand(-1, -1, latent_height // patch_size, latent_width // patch_size)
        .reshape(latent_timesteps.shape[0], -1)
    )


def load_frozen_encoders(checkpoint_dir: str | Path, wan_config, device: torch.device):
    """Load the existing Wan VAE and text encoder as immutable inference modules."""
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE

    checkpoint_dir = Path(checkpoint_dir)
    text_encoder = T5EncoderModel(
        text_len=wan_config.text_len,
        dtype=wan_config.t5_dtype,
        device=device,
        checkpoint_path=str(checkpoint_dir / wan_config.t5_checkpoint),
        tokenizer_path=wan_config.t5_tokenizer,
    )
    text_encoder.model.eval().requires_grad_(False)
    vae = Wan2_2_VAE(
        vae_pth=str(checkpoint_dir / wan_config.vae_checkpoint), device=device
    )
    vae.model.eval().requires_grad_(False)
    return vae, text_encoder


def load_trainable_dit(checkpoint_dir: str | Path, gradient_checkpointing: bool):
    """Load the pretrained TI2V DiT and enable its training-time checkpoint path."""
    from wan.modules.model import WanModel

    model = WanModel.from_pretrained(str(checkpoint_dir))
    model.train().requires_grad_(True)
    model.gradient_checkpointing = gradient_checkpointing
    return model
