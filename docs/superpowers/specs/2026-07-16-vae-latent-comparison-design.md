# VAE Latent Comparison Design

## Purpose

Add `compare_vae_latent.py`, a command-line utility that evaluates the Wan2.2 VAE reconstruction of an RGBA PNG frame sequence. It will encode and decode the sequence with `Wan2_2_VAE`, then write one labeled, side-by-side MP4: source/ground truth on the left and VAE reconstruction on the right.

## Scope

- Read `rgba_*.png` from a sample directory in natural numeric order.
- Default the sample directory to `examples/sample_0`; expose `--sample-path` to override it.
- Composite each RGBA frame over black, then pass three-channel RGB values normalized from `[0,1]` to `[-1,1]` to the VAE.
- Default the checkpoint to `./Wan2.2-TI2V-5B/Wan2.2_VAE.pth`, matching the normal README checkpoint-directory layout; expose `--vae-checkpoint` to override it.
- Preserve the sample’s 1280×704 resolution and 24 fps by default; expose `--fps` and `--output` overrides.
- Print MSE and PSNR for the normalized RGB source and reconstruction.
- Do not modify VAE code, run diffusion, train, or emit separate ground-truth/reconstruction videos.

## Design

### CLI

```text
python compare_vae_latent.py \
  --sample-path examples/sample_0 \
  --vae-checkpoint ./Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --output output/vae_comparison.mp4 \
  --fps 24
```

All options have the displayed defaults. The script fails clearly when no matching RGBA frames exist, input frames have inconsistent dimensions, a PNG lacks four channels, or the checkpoint is absent.

### Data flow

```text
rgba_00000.png ... rgba_00048.png
  -> numeric sort by suffix
  -> RGBA alpha compositing over black
  -> RGB tensor [3, T, H, W] in [-1, 1]
  -> Wan2_2_VAE.encode([video])
  -> Wan2_2_VAE.decode(latents)
  -> clamp/convert both clips to uint8 RGB
  -> per-frame GT | reconstruction panel with labels
  -> H.264 MP4 at configured fps
```

The supplied sample contains 49 frames, which matches the VAE’s `4n+1` temporal convention. The script accepts arbitrary non-empty frame counts but does not pad or trim them; VAE-compatible counts are the caller’s responsibility.

### Module boundaries

`compare_vae_latent.py` contains only focused helpers:

- `discover_rgba_frames(sample_path) -> list[Path]`: match and numeric-sort `rgba_*.png`.
- `load_rgb_video(frame_paths) -> torch.Tensor`: validate PNG structure, alpha-composite over black, and return `[3,T,H,W]` in `[-1,1]`.
- `reconstruct_video(vae, video) -> torch.Tensor`: call `vae.encode([video])` then `vae.decode(latents)` and return the sole reconstructed clip.
- `compute_metrics(gt, reconstruction) -> tuple[float,float]`: calculate RGB MSE and PSNR after matching dtype/device.
- `make_comparison_frame(gt_rgb, reconstruction_rgb) -> ndarray`: place equal-size uint8 RGB frames side by side and render fixed labels.
- `write_comparison_video(gt, reconstruction, output, fps)`: create the output directory, render frames in temporal order, and close the `imageio` writer even when writing fails.
- `main()`: parse CLI arguments, instantiate `Wan2_2_VAE`, coordinate the helpers, and print metrics/output location.

### Error handling

- `discover_rgba_frames` raises `FileNotFoundError` for no frames.
- `load_rgb_video` raises `ValueError` for non-RGBA images or inconsistent frame sizes.
- `main` raises `FileNotFoundError` before model construction if the checkpoint is absent.
- Output parent directories are created automatically.

### Testing

Add CPU-only pytest coverage in `tests/test_compare_vae_latent.py` using temporary RGBA images and a deterministic fake VAE. Tests verify numeric ordering, black alpha compositing, `[3,T,H,W]`/`[-1,1]` conversion, encode/decode invocation, side-by-side frame geometry, and a finite metric result. The tests do not load Wan checkpoints or allocate CUDA memory.

The real command is a separate manual integration check because it needs the checkpoint and a CUDA-capable runtime.
