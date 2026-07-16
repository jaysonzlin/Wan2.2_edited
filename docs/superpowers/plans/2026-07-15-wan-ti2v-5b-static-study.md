# Wan2.2 TI2V-5B Static Architecture Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a research-grade, line-referenced architecture dossier for Wan2.2-TI2V-5B without executing the model.

**Architecture:** Reconstruct the inference system in causal order: configuration and entry point, VAE representation, text conditioning, DiT forward pass, TI2V image conditioning, and flow-matching sampling. At every stage, retain a tensor ledger and separate verified code behavior from unverified training claims.

**Tech Stack:** Python source review, Markdown notes, Git; no CUDA execution, checkpoint loading, or internet retrieval in this plan.

## Global Constraints

- Static review only: do not run `generate.py`, import `torch`, load a checkpoint, or create GPU allocations.
- Cite code as `relative/path.py:line-line` for every factual architecture assertion.
- Treat `WanModel` constructor defaults as generic; use the downloaded checkpoint’s `config.json` when model-specific input/output dimensions matter.
- Do not infer training data, loss terms, or benchmark results from inference code.
- The standard 5B review case is 121 frames at 704×1280, using the values in `wan/configs/wan_ti2v_5B.py:15-36`.

---

## File structure

- Reference: `docs/superpowers/specs/2026-07-15-wan-ti2v-5b-static-study-design.md` — approved architecture-study design and evidence boundary.
- Create: `docs/research/wan-ti2v-5b-static-dossier.md` — personal, line-referenced architecture dossier produced by the study.
- Create: `docs/research/wan-ti2v-5b-claim-ledger.md` — table that distinguishes code-proven claims from claims requiring primary release/training material.
- Reference: `generate.py` — task routing.
- Reference: `wan/textimage2video.py` — unified T2V/I2V pipeline and sampler loop.
- Reference: `wan/modules/vae2_2.py` — causal high-compression VAE.
- Reference: `wan/modules/t5.py` — UMT5 encoder wrapper.
- Reference: `wan/modules/model.py` — DiT blocks, 3D RoPE, patching, and output head.
- Reference: `wan/utils/fm_solvers_unipc.py` and `wan/utils/fm_solvers.py` — flow-matching schedules and solvers.

### Task 1: Establish the model contract and tensor ledger

**Files:**
- Create: `docs/research/wan-ti2v-5b-static-dossier.md`
- Modify: none
- Test: manual algebra and source cross-check; no runtime test

**Interfaces:**
- Consumes: model configuration fields and `WanTI2V.t2v` latent-shape construction.
- Produces: a notation legend and a table with one row per representation, used by all following tasks.

- [x] **Step 1: Record the checkpoint-independent 5B contract**

Read `wan/configs/wan_ti2v_5B.py:11-36`. In the dossier, record these values with their code citations: UMT5 checkpoint name, VAE stride `(4,16,16)`, patch size `(1,2,2)`, DiT width `3072`, MLP width `14336`, `24` heads, `30` layers, global attention `(-1,-1)`, 50 steps, CFG `5.0`, and 121 frames at 24 fps.

- [x] **Step 2: Derive the standard latent shape on paper**

From `wan/textimage2video.py:283-291`, calculate:

```text
F_latent = (121 - 1) / 4 + 1 = 31
H_latent = 704 / 16 = 44
W_latent = 1280 / 16 = 80
patch grid = (31, 44 / 2, 80 / 2) = (31, 22, 40)
sequence length = 31 * 22 * 40 = 27,280
```

Add the result as a tensor-ledger row, noting that distributed sequence parallelism may pad the sequence to a multiple of `sp_size` (`wan/textimage2video.py:289-291`).

- [x] **Step 3: Capture configuration uncertainty explicitly**

Add a “checkpoint-only facts” subsection: `WanModel` defines defaults such as `in_dim=16` and `out_dim=16` at `wan/modules/model.py:305-320`, but `WanModel.from_pretrained(checkpoint_dir)` is used at `wan/textimage2video.py:102-109`. Therefore, defer exact DiT I/O-channel claims until reading the checkpoint’s `config.json` without loading weights.

- [x] **Step 4: Static validation**

Run:

```bash
rg -n "vae_stride|patch_size|frame_num|sample_fps" wan/configs/wan_ti2v_5B.py
rg -n "target_shape|seq_len" wan/textimage2video.py
```

Expected: the cited configuration values and shape equations are present at the referenced locations; do not run a Python program.

### Task 2: Document VAE representation and temporal causality

**Files:**
- Modify: `docs/research/wan-ti2v-5b-static-dossier.md`
- Test: line-reference audit; no runtime test

**Interfaces:**
- Consumes: tensor ledger from Task 1.
- Produces: a VAE subsection that explains latent normalization, compression, chunked causality, and the encode/decode contract.

- [x] **Step 1: Record the wrapper’s latent contract**

Read `wan/modules/vae2_2.py:888-1048`. State that `Wan2_2_VAE` constructs `WanVAE_` with `z_dim=48`, keeps channelwise mean/std statistics, encodes lists of video tensors, and decodes latent lists before clamping output into `[-1,1]`. Cite each assertion to these lines.

- [x] **Step 2: Trace encoder order**

Read `wan/modules/vae2_2.py:783-810`. Diagram this exact sequence:

```text
input video -> patchify(patch_size=2) -> causal Encoder3d chunks
            -> conv1 -> split(mu, log_var) -> affine channel normalization -> latent
```

Document that normal inference returns `mu`; do not confuse the unused `log_var` branch with a sampled latent.

- [x] **Step 3: Trace decoder order and cache behavior**

Read `wan/modules/vae2_2.py:812-860`. Diagram inverse normalization, `conv2`, one-slice-at-a-time decoder calls, cached causal convolutions, and final `unpatchify(patch_size=2)`. Add the question “what is the receptive-field implication of cached causal convolutions?” rather than claiming a receptive field without deriving kernel details.

- [x] **Step 4: Static validation**

Run:

```bash
rg -n "def encode|def decode|patchify|unpatchify|log_var|clear_cache" wan/modules/vae2_2.py
```

Expected: each VAE-diagram operation has a matching source location.

### Task 3: Document text conditioning and one DiT block

**Files:**
- Modify: `docs/research/wan-ti2v-5b-static-dossier.md`
- Test: equation-to-code audit; no runtime test

**Interfaces:**
- Consumes: Task 1 token notation and Task 2 latent symbols.
- Produces: text-context notation and a block-level equation sheet.

- [x] **Step 1: Record UMT5 encoder behavior**

Read `wan/modules/t5.py:456-513` and `wan/configs/shared_config.py:7-12`. Record the UMT5-XXL encoder dimensions (4096 width, 24 encoder layers, 64 heads) and that the wrapper tokenizes, computes unpadded lengths, and returns context sequences trimmed to those lengths.

- [x] **Step 2: Trace DiT input preparation**

Read `wan/modules/model.py:377-405` and `wan/modules/model.py:444-487`. In the dossier, identify the three distinct embeddings:

```text
latent volume -> Conv3d patch_embedding -> video tokens
UMT5 context -> text_embedding MLP -> DiT-width text tokens
timestep -> sinusoidal_embedding_1d -> time_embedding -> six per-token modulation vectors
```

State that context is padded to the fixed text length only after the UMT5 wrapper has removed original padding (`wan/modules/model.py:471-478`, `wan/modules/t5.py:506-513`).

- [x] **Step 3: Write the block equation sheet**

Read `wan/modules/model.py:183-259`. Define the six chunks from `e0` as `(s_sa, a_sa, g_sa, s_mlp, a_mlp, g_mlp)` in the order used by code. Write:

```text
h = x + g_sa ⊙ SA(LN(x) ⊙ (1 + a_sa) + s_sa)
x' = h + CrossAttn(Norm3(h), context)
y = x' + g_mlp ⊙ MLP(LN(x') ⊙ (1 + a_mlp) + s_mlp)
```

Include the qualification that this is notation for the code’s operations; `Norm3` is a learnable LayerNorm only when `cross_attn_norm=True` (`wan/modules/model.py:206-210`).

- [x] **Step 4: Add positional and output pathways**

Read `wan/modules/model.py:30-65`, `wan/modules/model.py:88-151`, and `wan/modules/model.py:489-522`. Explain that self-attention applies RMS-normalized Q/K and three-axis RoPE, while cross-attention does not call `rope_apply`; then document the modulated head and unpatchify process.

- [x] **Step 5: Static validation**

Run:

```bash
rg -n "time_projection|text_embedding|class WanAttentionBlock|rope_apply|def unpatchify" wan/modules/model.py
```

Expected: every equation term maps to an implementation symbol.

### Task 4: Explain why TI2V supports both T2V and I2V

**Files:**
- Modify: `docs/research/wan-ti2v-5b-static-dossier.md`
- Modify: `docs/research/wan-ti2v-5b-claim-ledger.md`
- Test: branch comparison; no runtime test

**Interfaces:**
- Consumes: VAE and DiT contracts from Tasks 2–3.
- Produces: a line-referenced T2V/I2V comparison and an explicit evidence-boundary ledger.

- [x] **Step 1: Document task routing**

Read `generate.py:428-454` and `wan/textimage2video.py:162-237`. State that the same `WanTI2V` object branches into `i2v` when `img is not None` and `t2v` otherwise—not into separate model classes.

- [x] **Step 2: Contrast initial states**

Read `wan/textimage2video.py:311-320` and `wan/textimage2video.py:461-512`. Make a two-column table:

| T2V | I2V |
|---|---|
| Initializes all latent cells from seeded Gaussian noise. | Resizes/crops the image to latent-compatible dimensions, VAE-encodes it, and initializes the first latent-time slice from that encoding. |

Add exact code citations in each cell.

- [x] **Step 3: Explain protection of the input image during sampling**

Read `wan/textimage2video.py:548-600` and `wan/utils/utils.py:172-199`. Explain the mask construction and the two enforcement sites: timestep zero at first-frame patch positions (`wan/textimage2video.py:573-578`) and latent re-imposition after each scheduler step (`wan/textimage2video.py:591-598`). Do not label this “inpainting” unless a primary source uses that term.

- [x] **Step 4: Populate evidence boundaries**

In `docs/research/wan-ti2v-5b-claim-ledger.md`, add this table:

| Claim | Status | Evidence | Next required source |
|---|---|---|---|
| T2V and I2V use one pipeline class and DiT checkpoint. | Code-proven | `generate.py:428-454`; `wan/textimage2video.py:102-109,162-237` | None |
| Image conditioning preserves the first latent frame each step. | Code-proven | `wan/textimage2video.py:548-600` | None |
| The model’s training objective is flow matching. | Partially supported | Scheduler config says `flow_prediction` at `wan/utils/fm_solvers_unipc.py:22-46`. | Training paper/report or training code |
| Exact checkpoint channel dimensions. | Unverified | Generic constructor defaults are at `wan/modules/model.py:305-320`. | Checkpoint `config.json` |

- [x] **Step 5: Static validation**

Run:

```bash
rg -n "def t2v|def i2v|mask2|latent = \(1\. - mask2|noise_pred_cond" wan/textimage2video.py
rg -n "def masks_like" wan/utils/utils.py
```

Expected: the branch, mask, and CFG claims each have source evidence.

### Task 5: Reconstruct sampling semantics and close the dossier

**Files:**
- Modify: `docs/research/wan-ti2v-5b-static-dossier.md`
- Modify: `docs/research/wan-ti2v-5b-claim-ledger.md`
- Test: claim-reference review; no runtime test

**Interfaces:**
- Consumes: conditional/unconditional context notation from Tasks 3–4.
- Produces: a complete, source-grounded static dossier and prioritized future questions.

- [x] **Step 1: Record CFG as an equation**

Read `wan/textimage2video.py:380-394`. Define `v_c` and `v_u` as the two DiT outputs and record:

```text
v_cfg = v_u + guidance_scale * (v_c - v_u)
```

Document that the default scale is `5.0` in `wan/configs/wan_ti2v_5B.py:31-36` and that the negative prompt is pulled from shared configuration when no negative prompt is supplied (`wan/textimage2video.py:293-295`).

- [x] **Step 2: Record the sampling schedule contract**

Read `wan/utils/fm_solvers_unipc.py:22-46` and `wan/utils/fm_solvers_unipc.py:161-215`. State only that the default scheduler is adapted for `flow_prediction`, configures 1,000 training timesteps, and applies the rational shift `shift * sigma / (1 + (shift - 1) * sigma)`. Cite the exact lines.

- [x] **Step 3: Add solver boundary notes**

Read `wan/textimage2video.py:335-354` and `wan/utils/fm_solvers.py:227-281`. Record that the pipeline exposes UniPC and DPM++ branches, but defer claims about quality or stability tradeoffs to later controlled experiments.

- [x] **Step 4: Perform the final claim audit**

For every sentence in the dossier and claim ledger, confirm one of two outcomes:

```text
1. It ends with a code citation that directly establishes it.
2. The ledger marks it as an open question and names the required primary source.
```

Run:

```bash
rg -n "TBD|TODO|probably|should|training data|benchmark" docs/research/wan-ti2v-5b-static-dossier.md docs/research/wan-ti2v-5b-claim-ledger.md
```

Expected: no placeholders; any training or benchmark wording is explicitly scoped as unknown or deferred.

### Task 6: Plan the post-static research phase without executing it

**Files:**
- Modify: `docs/research/wan-ti2v-5b-claim-ledger.md`
- Test: each future experiment traces to one static hypothesis; no runtime test today

**Interfaces:**
- Consumes: completed dossier and claim ledger.
- Produces: a priority-ordered, hypothesis-driven experimental backlog for later H200 use.

- [x] **Step 1: Add checkpoint metadata review as the first follow-up**

Add the non-execution action: inspect the downloaded checkpoint’s JSON configuration and weight index. Its output is an update to the tensor ledger with verified DiT input/output dimensions and model type.

- [x] **Step 2: Add primary-source reconciliation as the second follow-up**

Add the non-execution action: read the Wan2.2 primary technical material and map its claims about training, objective, and architecture to the dossier. Any mismatch becomes a separate ledger item rather than an edit made without a source.

- [x] **Step 3: Add H200 experiments in dependency order**

Add this future sequence, each gated on a written hypothesis from the static dossier:

```text
VAE encode/decode round trip
-> T2V versus I2V first-frame/mask ablation
-> CFG and shift sweep
-> UniPC versus DPM++ comparison
-> activation and attention instrumentation
```

- [x] **Step 4: Static validation**

Read the backlog and confirm each item tests a named static claim. If an experiment merely “explores quality,” rewrite it as a hypothesis test before authorizing GPU work.

## Plan self-review

- **Spec coverage:** Tasks 1–5 cover configuration, VAE, text conditioning, DiT, TI2V routing/masking, CFG, and flow scheduling. Task 6 preserves the later research path without violating today’s static-only constraint.
- **Evidence discipline:** every task names exact source locations and adds unknowns to the claim ledger instead of guessing.
- **Scope:** no code modifications to the model, no checkpoint execution, and no experiments are included.
