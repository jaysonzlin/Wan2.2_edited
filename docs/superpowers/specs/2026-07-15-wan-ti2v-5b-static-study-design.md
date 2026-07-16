# Wan2.2 TI2V-5B Static Architecture Study Design

## Purpose

Develop research-level, evidence-backed understanding of the Wan2.2-TI2V-5B inference architecture without running the model today. Every technical conclusion must cite the implementation file and line(s) from which it follows. Do not treat an inference implementation as evidence of training data, training loss, or evaluation methodology.

## Scope and constraints

- **Target:** the dense, unified `Wan2.2-TI2V-5B` checkpoint.
- **Today:** read code only; no checkpoint loading, inference, benchmarks, or ablations.
- **Background:** the reader is comfortable with PyTorch and diffusion transformers.
- **Available later:** multiple H200 GPUs, for validation work after the static model is reconstructed.

## Architectural hypothesis to validate from code

```text
prompt ──> UMT5-XXL encoder ──> projected text context ┐
                                                        │
noise or image-conditioned latent ─> patchified DiT ───┼─> flow field
                                                        │      │
per-token timestep modulation + 3D RoPE ──────────────┘      v
                                                     flow-matching scheduler
                                                               │
                                                         VAE decoder
                                                               │
                                                             video
```

The entry point selects `WanTI2V` for the `ti2v-5B` task and passes its sampling arguments through unchanged (`generate.py:428-454`). `WanTI2V.generate` selects image-to-video only when an image is supplied; otherwise it takes the text-to-video path (`wan/textimage2video.py:162-237`).

## Review sequence

### 1. Model contract and tensor ledger (20 minutes)

Read the 5B configuration first. It fixes VAE stride `(4, 16, 16)`, spatial patch size `(1, 2, 2)`, transformer width 3072, 24 heads, 30 layers, and default sampling settings (`wan/configs/wan_ti2v_5B.py:15-36`).

For a 121-frame, 704×1280 clip, derive these spatial-temporal dimensions:

| Stage | Shape / token count | Evidence |
|---|---:|---|
| VAE latent grid | `31 × 44 × 80` | `((121 - 1) / 4) + 1`, `704 / 16`, and `1280 / 16`; the T2V code constructs this exact shape from `vae_stride` (`wan/textimage2video.py:283-287`). |
| Patch grid | `31 × 22 × 40` | DiT patch embedding has stride `patch_size`; the configuration sets `(1,2,2)` (`wan/modules/model.py:377-379`, `wan/configs/wan_ti2v_5B.py:20`). |
| DiT sequence | `27,280` tokens | `31 × 22 × 40`; the inference path computes padded sequence length from this grid (`wan/textimage2video.py:289-291`). |

**Open verification item:** inspect the checkpoint's `config.json` before asserting its input/output channel counts. `WanModel` source defaults are generic values (`wan/modules/model.py:305-320`); the checkpoint configuration is authoritative for its instantiated dimensions.

### 2. Video VAE and latent representation (45 minutes)

Read `WanVAE_` before the transformer. The wrapper creates a 48-dimensional latent VAE (`wan/modules/vae2_2.py:888-1022`). Encoding patchifies RGB video by 2 before applying the causal 3D encoder, processes temporal chunks, takes the mean branch of a two-way latent projection, then normalizes it with fixed channel statistics (`wan/modules/vae2_2.py:783-810`). Decoding reverses fixed-channel normalization and decodes one latent temporal slice at a time with convolution caches (`wan/modules/vae2_2.py:812-839`).

Answer in notes:

1. How does pixel patchification combine with the two enabled temporal downsample stages to produce the claimed `4×16×16` VAE stride?
2. What temporal information can or cannot flow through causal convolutions and their caches?
3. Why does inference use `mu` rather than sampling from `(mu, log_var)`?

### 3. Text conditioning (30 minutes)

The text encoder is an encoder-only UMT5-XXL: width 4096, 24 encoder layers, 64 heads, and maximum token length 512 (`wan/modules/t5.py:456-469`; `wan/configs/shared_config.py:7-12`). Its wrapper tokenizes, finds the unpadded lengths, runs the encoder, and returns each sequence without trailing pads (`wan/modules/t5.py:506-513`). The DiT re-pads each context to its fixed text length and maps it from text width to DiT width through a two-layer MLP (`wan/modules/model.py:471-478`; `wan/modules/model.py:380-382`).

State precisely whether an assertion describes UMT5, the adapter, or the cross-attention mechanism—these are distinct components.

### 4. DiT core (60 minutes)

Trace the `WanModel.forward` method end to end.

- Latent inputs are 3D-convolved into patch tokens, flattened, and right-padded to the batch sequence length (`wan/modules/model.py:444-457`).
- Timesteps receive sinusoidal embeddings, then a SiLU-MLP creates six modulation vectors for every token (`wan/modules/model.py:459-469`).
- Each of the 30 blocks performs time-modulated self-attention, text cross-attention, and a time-modulated GELU MLP (`wan/modules/model.py:183-259`).
- Self-attention applies Q/K RMS normalization and 3D RoPE before Flash Attention (`wan/modules/model.py:88-151`). RoPE divides head dimensions across temporal, height, and width axes (`wan/modules/model.py:30-65`).
- The head applies final modulation, projects back into patch values, and `unpatchify` restores the latent volume (`wan/modules/model.py:262-291`, `wan/modules/model.py:489-522`).

For one block, write equations for all six modulation branches and label their roles: self-attention shift, scale, gate; MLP shift, scale, gate. Separately document that cross-attention is un-gated in this implementation (`wan/modules/model.py:250-256`).

### 5. Unified text-to-video and image-to-video conditioning (45 minutes)

T2V starts from Gaussian latent noise (`wan/textimage2video.py:311-320`). I2V preprocesses the input to compatible dimensions, encodes it with the same VAE, inserts the encoded first frame into the initial latent state, and re-imposes that image-derived latent after every scheduler step (`wan/textimage2video.py:461-512`, `wan/textimage2video.py:548-600`).

The I2V timestep sequence assigns zero to the spatial positions belonging to the protected first latent frame and the current timestep elsewhere (`wan/textimage2video.py:573-578`). This is the key static explanation of TI2V unification: image conditioning is expressed in the latent/timestep construction rather than through a separate model class or additional image cross-attention pathway.

### 6. Flow-matching sampling and classifier-free guidance (30 minutes)

For every timestep, the code evaluates conditional and negative-prompt contexts separately, then combines their predicted flow fields using classifier-free guidance (`wan/textimage2video.py:380-394`; `wan/textimage2video.py:580-597`). The default UniPC scheduler declares `prediction_type="flow_prediction"` (`wan/utils/fm_solvers_unipc.py:22-46`) and applies the configurable rational shift to the sigma schedule (`wan/utils/fm_solvers_unipc.py:161-215`). Therefore, describe the DiT output as a flow prediction unless later training documentation establishes a more precise parameterization.

## End-of-day deliverable

Create a two-page personal architecture note with:

1. The tensor ledger above, completed with checkpoint-derived channel dimensions.
2. A one-page forward-pass diagram using only verified code paths.
3. A block-level equation sheet.
4. A T2V versus I2V comparison, line-referenced.
5. A two-column list: **known from inference code** versus **requires source paper, technical report, checkpoint metadata, or training code**.

## Follow-on research plan (not today)

1. **Documentation reconciliation:** obtain primary release/training material and map every claim to the code-level reconstruction. Resolve objective, data, and loss-function questions rather than inferring them from the scheduler.
2. **Checkpoint inspection:** review checkpoint configuration and weight tensor names to close model-dimension and component-boundary gaps.
3. **Controlled validation on H200s:** only after the static note is complete, run VAE round-trip studies; T2V/I2V mask and timestep ablations; CFG, shift, and solver sweeps; then instrument activations/attention. Each experiment should test a written static hypothesis.
4. **Research synthesis:** maintain a claim ledger with source, confidence, relevant code lines, experiment, and conclusion.

## Static-review quality checks

- Every architecture assertion includes a file and exact line range.
- No claim about training data, losses, or empirical quality is asserted from this repository alone.
- Generic source defaults are not conflated with checkpoint-specific configuration.
- T2V and I2V are explained as separate pipeline paths that share one DiT/VAE stack.
