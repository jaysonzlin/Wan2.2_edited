# Wan2.2 TI2V-5B Static Architecture Dossier

## 1. Model contract and tensor ledger

### 1.1 Checkpoint-independent 5B configuration

The TI2V-5B configuration selects the UMT5-XXL encoder checkpoint `models_t5_umt5-xxl-enc-bf16.pth` (`wan/configs/wan_ti2v_5B.py:11-13`). Its VAE checkpoint is `Wan2.2_VAE.pth` and its nominal video-to-latent stride is `(T,H,W) = (4,16,16)` (`wan/configs/wan_ti2v_5B.py:15-17`).

The DiT uses spatial-only patching `(1,2,2)`, width 3072, MLP width 14336, 24 attention heads, 30 blocks, and global self-attention (`window_size=(-1,-1)`) (`wan/configs/wan_ti2v_5B.py:19-29`). The default inference contract is 24 fps, 121 frames, 50 sampling steps, shift 5.0, and CFG scale 5.0 (`wan/configs/wan_ti2v_5B.py:31-36`).

### 1.2 Standard 720p tensor ledger

This ledger follows the default 121-frame 1280×704 T2V path. `WanTI2V.t2v` constructs its initial latent shape from the VAE’s runtime `z_dim`, the frame count, and `vae_stride` (`wan/textimage2video.py:283-287`); it derives DiT sequence length from the latent spatial dimensions and the spatial components of `patch_size` (`wan/textimage2video.py:289-291`).

| Representation | Shape | Derivation / evidence |
|---|---|---|
| Output video | `[3, 121, 704, 1280]` | Standard review case: the TI2V size allowlist includes `1280*704` (`wan/configs/__init__.py:43-49`) and config uses 121 frames (`wan/configs/wan_ti2v_5B.py:36`). The VAE decoder produces 12 patchified channels before the final 2× unpatchify, yielding RGB (`wan/modules/vae2_2.py:640-669`, `wan/modules/vae2_2.py:812-839`). |
| VAE latent volume | `[48, 31, 44, 80]` | `F_latent=(121-1)//4+1=31`, `H_latent=704//16=44`, `W_latent=1280//16=80` from the construction at `wan/textimage2video.py:283-287`. The VAE wrapper’s default `z_dim` is 48 (`wan/modules/vae2_2.py:888-894`). |
| Patch grid | `[31, 22, 40]` | The patch shape `(1,2,2)` preserves latent time and halves both latent spatial axes (`wan/configs/wan_ti2v_5B.py:19-20`; `wan/modules/model.py:377-379`). |
| Unpadded DiT token count | `31 × 22 × 40 = 27,280` | The pipeline divides the latent spatial area by `patch_size[1] × patch_size[2]` and multiplies by latent temporal length (`wan/textimage2video.py:289-291`). |
| DiT hidden tokens | `[1, 27,280, 3072]` before optional sequence-parallel padding | Patch embeddings are flattened to token sequences (`wan/modules/model.py:448-457`); width is 3072 (`wan/configs/wan_ti2v_5B.py:21`). |

With sequence parallelism, `seq_len` is rounded up to a multiple of `sp_size`, so the allocated token sequence can be larger than 27,280 (`wan/textimage2video.py:289-291`).

### 1.3 Checkpoint-specific facts not yet asserted

`WanModel` has generic source defaults `in_dim=16` and `out_dim=16` (`wan/modules/model.py:305-320`), but TI2V constructs the DiT with `WanModel.from_pretrained(checkpoint_dir)` (`wan/textimage2video.py:102-109`). Therefore this dossier does not use those generic defaults to describe the 5B checkpoint. The VAE-facing latent is 48-channel by wrapper configuration; confirm the checkpoint’s DiT input/output channel fields from its `config.json` before treating that interface as checkpoint-verified.

## 2. High-compression causal VAE

### 2.1 Compression accounting

The VAE wrapper instantiates `WanVAE_` with `z_dim=48`, `dim_mult=[1,2,4,4]`, and enabled temporal downsampling only in the second and third encoder downsample blocks: `[False,True,True]` (`wan/modules/vae2_2.py:888-1022`).

Before the encoder, `patchify(x, patch_size=2)` preserves the temporal axis, halves height and width, and packs each 2×2 RGB neighborhood into channels: `[B,3,T,H,W] → [B,12,T,H/2,W/2]` (`wan/modules/vae2_2.py:280-296`; `wan/modules/vae2_2.py:783-786`). `Encoder3d` creates four residual stages, but only the first three have `down_flag=True` (`wan/modules/vae2_2.py:527-543`). Consequently those three stages each halve spatial resolution. The configured temporal flags mean stage 1 is 2D-only while stages 2 and 3 use 3D downsampling (`wan/modules/vae2_2.py:530-543`; `wan/modules/vae2_2.py:888-897`).

The resulting stride is therefore:

```text
temporal: 1 × 2 × 2 = 4
spatial: 2 (initial patchify) × 2 × 2 × 2 = 16
```

This establishes the wrapper’s stated `4×16×16` latent stride from code rather than from a release claim. For the standard clip, it maps `[3,121,704,1280]` to `[48,31,44,80]`.

### 2.2 Encoder contract

`WanVAE_.encode` processes the patchified video in an initial one-frame chunk followed by four-frame chunks, while preserving feature caches across chunks (`wan/modules/vae2_2.py:783-802`). The encoder returns twice the latent channel count, after which `conv1` splits that tensor into `mu` and `log_var`; inference applies the fixed per-channel affine normalization only to `mu` and returns it (`wan/modules/vae2_2.py:803-810`). Although helper methods can reparameterize with `log_var`, the wrapper’s production `encode` path invokes `self.model.encode` and receives the returned deterministic `mu` (`wan/modules/vae2_2.py:841-851`; `wan/modules/vae2_2.py:1024-1033`).

The causal property is implemented in `CausalConv3d`: temporal padding is placed entirely on the left and never on the right (`wan/modules/vae2_2.py:17-42`). At chunk boundaries, residual/encoder convolutions cache their final two feature-time slices and prepend the cached past to the next chunk (`wan/modules/vae2_2.py:214-235`; `wan/modules/vae2_2.py:559-613`). Therefore a latent position can depend on preceding video content but not future video content through these convolutions. The VAE’s middle attention is spatial per frame—not temporal—because it folds time into the batch dimension before `scaled_dot_product_attention` (`wan/modules/vae2_2.py:255-277`). The exact temporal receptive field remains a derived quantity: it requires counting every causal convolution and downsampler rather than merely observing that caching exists.

### 2.3 Decoder contract

Decoding first reverses the per-channel latent normalization, then applies `conv2` and decodes one latent-time slice at a time while maintaining decoder caches (`wan/modules/vae2_2.py:812-839`). The decoder receives the reversed temporal-upsample flags from `WanVAE_` (`wan/modules/vae2_2.py:753-775`), mirrors the spatial upsample stages (`wan/modules/vae2_2.py:649-669`), and finishes with 12 channels that `unpatchify(..., patch_size=2)` converts back into three RGB channels (`wan/modules/vae2_2.py:665-669`; `wan/modules/vae2_2.py:837-839`; `wan/modules/vae2_2.py:299-313`).

The first decoder chunk removes `factor_t - 1` leading positions after duplication, avoiding an extra initial temporal output caused by upsampling (`wan/modules/vae2_2.py:390-412`; `wan/modules/vae2_2.py:821-829`).

## 3. Text conditioning and DiT computation

### 3.1 Text encoder and adapter

The configured text encoder is encoder-only UMT5-XXL with vocabulary 256,384, hidden width 4096, 64 heads, 24 encoder layers, and feed-forward width 10,240 (`wan/modules/t5.py:456-469`). TI2V fixes the tokenizer context budget to 512 tokens (`wan/configs/shared_config.py:8-12`).

`T5EncoderModel.__call__` tokenizes text, uses the token mask to compute each unpadded length, runs the UMT5 encoder, and returns each context tensor truncated to that unpadded length (`wan/modules/t5.py:506-513`). `WanModel.forward` then right-pads each variable-length UMT5 output to `text_len=512` and maps it through `Linear(4096,3072) → GELU(tanh) → Linear(3072,3072)` (`wan/modules/model.py:377-382`; `wan/modules/model.py:471-478`). Thus text conditioning is a separate encoder plus adapter, rather than tokens embedded directly in the video transformer.

One source-level subtlety: `WanModel.forward` explicitly sets `context_lens = None` before cross-attention (`wan/modules/model.py:471-487`). In `flash_attention`, a `None` key-length argument makes all key/value positions participate (`wan/modules/attention.py:71-80`). Therefore the DiT cross-attention receives a fixed 512-position context, including the adapter outputs at re-padded positions. This is a code observation; whether the checkpoint was trained to exploit or neutralize those positions requires training material or a later controlled probe.

### 3.2 Video, time, and position representations

The DiT converts each latent volume with a stride-equals-kernel `Conv3d` patch embedding (`wan/modules/model.py:377-379`). It then flattens the patch grid into tokens, records each unpadded video length, and pads video tokens only to the batch’s maximum sequence length (`wan/modules/model.py:448-457`). For the single 720p example, this is the 27,280-token sequence from Section 1.

For a timestep tensor `t`, the model first obtains a 256-dimensional sinusoidal embedding, applies `Linear(256,3072) → SiLU → Linear(3072,3072)`, and projects that result to six 3072-dimensional vectors per video token (`wan/modules/model.py:14-24`; `wan/modules/model.py:384-386`; `wan/modules/model.py:459-469`). The timestep path is per-token even in T2V; that generality is what permits I2V to use different timestep values for image-derived and generated latent positions later in the pipeline.

With width 3072 and 24 heads, each attention head has 128 real dimensions. The model builds 3D RoPE frequencies from that head width (`wan/modules/model.py:397-405`). `rope_apply` divides the 64 complex dimensions into temporal, height, and width pieces of 22, 21, and 21 complex dimensions—equivalently 44, 42, and 42 real dimensions—then applies the corresponding position frequency over the `(F,H,W)` token grid (`wan/modules/model.py:38-66`).

### 3.3 One Wan attention block

Let `E = time_projection(time_embedding(t))`, and let a block add its learned `modulation` parameter before splitting `E` into six per-token vectors. The addition and split occur at `wan/modules/model.py:237-240`. Name the vectors in code order:

```text
(s_sa, a_sa, g_sa, s_mlp, a_mlp, g_mlp)
```

The block’s equations, directly corresponding to `wan/modules/model.py:242-258`, are:

```text
u   = LN1(x) ⊙ (1 + a_sa) + s_sa
h   = x + g_sa ⊙ SelfAttn(u)
r   = h + CrossAttn(Norm3(h), text_context)
out = r + g_mlp ⊙ MLP(LN2(r) ⊙ (1 + a_mlp) + s_mlp)
```

`LN1` and `LN2` are non-affine `WanLayerNorm` instances, while `Norm3` is an affine LayerNorm because TI2V enables `cross_attn_norm=True` (`wan/modules/model.py:88-98`; `wan/modules/model.py:203-214`; `wan/configs/wan_ti2v_5B.py:27-29`). Cross-attention itself has no separate time-controlled gate: its output is added directly before the MLP (`wan/modules/model.py:250-256`).

Self-attention applies Q and K projections followed by RMS normalization, reshapes into heads, applies 3D RoPE to Q/K, and calls Flash Attention with the true video sequence lengths (`wan/modules/model.py:118-155`). Cross-attention uses the same Q/K normalizers but omits RoPE and uses text context as K/V (`wan/modules/model.py:158-180`).

### 3.4 Output head

After all blocks, the model applies a final two-vector time-modulated LayerNorm/head projection, where the patch output dimension is `out_dim × product(patch_size)` (`wan/modules/model.py:262-291`). `unpatchify` discards batch padding, reshapes valid token predictions into the `(F,H,W)` patch grid, permutes patch axes, and restores the latent volume (`wan/modules/model.py:489-522`).

## 4. Unified text-to-video and image-to-video conditioning

### 4.1 One pipeline object, two initial-state constructions

The CLI routes `ti2v-5B` to a single `WanTI2V` object (`generate.py:428-454`). `WanTI2V.generate` chooses `i2v` whenever `img is not None` and otherwise chooses `t2v` (`wan/textimage2video.py:162-237`). Both paths use the same VAE, text encoder, loaded DiT checkpoint, scheduler choices, and CFG computation; their architectural difference is the initial latent and the per-token timestep vector.

For T2V, the initial state is one seeded Gaussian tensor with shape `[z_dim,F_latent,H_latent,W_latent]` (`wan/textimage2video.py:283-320`). `masks_like(..., zero=False)` returns all-one masks, so the T2V timestep vector contains the same current scheduler timestep at all valid video-token positions (`wan/textimage2video.py:356-378`; `wan/utils/utils.py:172-199`).

For I2V, the source image is resized/cropped to dimensions divisible by `patch_size × vae_stride = 32` in each spatial axis, transformed to `[-1,1]`, and given a singleton temporal axis before VAE encoding (`wan/textimage2video.py:461-483`). The resulting image latent has one temporal slice. The generation noise has the same full video-latent shape as T2V (`wan/textimage2video.py:488-494`).

### 4.2 Initial-latent construction and persistent anchoring

In the no-generator branch used by inference, `masks_like([noise], zero=True)` sets `mask2[:,0]` to zero and leaves all other latent-time positions as one (`wan/utils/utils.py:172-199`). Let `z_img` be the one-slice VAE encoding of the input image and `ε` be full-video noise. The initial I2V latent is exactly:

```text
x_t[0] = (1 - M) ⊙ z_img + M ⊙ ε,
where M[:, 0, :, :] = 0 and M[:, τ>0, :, :] = 1.
```

This is implemented at `wan/textimage2video.py:548-552`. The same equation is executed again after every scheduler update (`wan/textimage2video.py:591-598`). Consequently, the first **VAE latent-time slice** is kept equal to the encoded image throughout the entire denoising trajectory; it is more precise to name this latent slice than to call the operation generic image conditioning.

The causal decoder turns the first latent slice into its special first output chunk, then produces later chunks from later latent slices (`wan/modules/vae2_2.py:812-839`; `wan/modules/vae2_2.py:390-412`). This is consistent with the protected initial latent slice anchoring the initial decoded frame, but the exact pixel-level fidelity and later-frame influence remain empirical questions.

### 4.3 Per-token timestep conditioning separates known from generated content

Immediately before each DiT call, I2V takes the first channel’s mask, subsamples it spatially by the DiT patch factors, multiplies it by the current scheduler timestep, and flattens it into the token timestep vector (`wan/textimage2video.py:567-578`). On the standard grid, this yields 0 for all `22×40` patch tokens at latent time 0 and the current timestep for all later video tokens:

```text
t_token[f=0, h, w] = 0
t_token[f>0, h, w] = t_scheduler
```

The spatial `::2` slicing matches the DiT’s `(1,2,2)` patching: it maps the `[31,44,80]` VAE mask to the `[31,22,40]` DiT grid (`wan/textimage2video.py:573-578`; `wan/configs/wan_ti2v_5B.py:19-20`). This explains why the DiT was built with a per-token timestep input rather than only a scalar diffusion step.

**Static conclusion:** TI2V unification is implemented by a shared latent DiT/VAE stack plus an I2V-specific initial latent, invariant mask, and per-token timestep map. The code does not introduce a separate image cross-attention encoder or a second I2V DiT class in this pathway.

## 5. Classifier-free guidance and flow-matching solvers

### 5.1 CFG uses a negative-prompt baseline

Before sampling, the pipeline encodes the user prompt and a second prompt named `n_prompt` (`wan/textimage2video.py:299-309`). If no `n_prompt` is provided, it is replaced by the configured Chinese negative-prompt string (`wan/textimage2video.py:293-295`; `wan/configs/shared_config.py:16-19`). The source variable is called `noise_pred_uncond`, but in the default configuration it is more accurately a **negative-prompt-conditioned** prediction rather than a prediction from an empty text context.

At each scheduler timestep, the DiT runs once with the user-prompt context and once with the negative-prompt context, then combines the results as:

```text
v_cfg = v_neg + g × (v_pos - v_neg),
```

where `g` is `guide_scale` (`wan/textimage2video.py:360-386`). The TI2V-5B default is `g = 5.0` (`wan/configs/wan_ti2v_5B.py:31-35`). This guidance happens on the DiT’s latent-space output before the scheduler step; the VAE only runs after the final latent is obtained (`wan/textimage2video.py:388-401`).

### 5.2 Shifted flow schedule

The default branch constructs `FlowUniPCMultistepScheduler` with 1,000 nominal training timesteps and passes the TI2V shift value to `set_timesteps` (`wan/textimage2video.py:335-342`; `wan/configs/shared_config.py:16-17`). UniPC begins from a linear sigma grid and applies the rational shift:

```text
sigma' = shift × sigma / (1 + (shift - 1) × sigma).
```

The schedule implementation is at `wan/utils/fm_solvers_unipc.py:184-213`. It then reports the model timestep as `sigma' × 1000`, stored as `int64` (`wan/utils/fm_solvers_unipc.py:207-215`). The final sigma is explicitly appended as zero (`wan/utils/fm_solvers_unipc.py:197-209`).

For the alternative `dpm++` path, the pipeline explicitly builds a linearly spaced `[1,0)` sigma sequence and applies the same rational shift before handing it to `FlowDPMSolverMultistepScheduler` (`wan/textimage2video.py:343-352`; `wan/utils/fm_solvers.py:24-28`). Thus `shift` changes the inference trajectory in both exposed solver branches; code alone does not establish an optimal value or quality trade-off.

### 5.3 What “flow prediction” means in this inference code

Both solver classes configure `prediction_type="flow_prediction"` (`wan/utils/fm_solvers_unipc.py:79-96`; `wan/utils/fm_solvers.py:127-147`). With UniPC’s default `predict_x0=True`, the scheduler converts a DiT output `v` and current latent `x_sigma` to:

```text
x0_pred = x_sigma - sigma × v.
```

This conversion is explicit at `wan/utils/fm_solvers_unipc.py:317-333`. Therefore the most precise inference-only description is: **the DiT output is consumed as a flow prediction from which the scheduler derives an x0 prediction.** Rearranging the implemented equation gives `v = (x_sigma - x0_pred) / sigma`.

That establishes the sampler interface—not the full training objective, loss weighting, noise/data interpolation procedure, or the provenance of model weights. Those still require primary Wan2.2 training material or training code.

### 5.4 Solver update behavior

`FlowUniPCMultistepScheduler.step` first converts the raw DiT output, then after the first step may apply its UniC corrector, retains recent model outputs/timesteps, and uses a warm-started multistep UniP update (`wan/utils/fm_solvers_unipc.py:657-741`). The default solver order is two (`wan/utils/fm_solvers_unipc.py:79-96`), so it transitions from first-order warmup to a second-order multistep method when enough history is available (`wan/utils/fm_solvers_unipc.py:714-733`).

`FlowDPMSolverMultistepScheduler` also converts a flow prediction to `x0_pred` in its DPM++ configuration (`wan/utils/fm_solvers.py:381-395`) and selects first-, second-, or third-order updates according to solver order and available history (`wan/utils/fm_solvers.py:744-799`). Neither implementation introduces an extra learned network; they are numerical solvers around the same guided DiT output.
