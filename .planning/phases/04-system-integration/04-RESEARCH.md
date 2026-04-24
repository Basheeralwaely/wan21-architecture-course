# Phase 4: System Integration - Research

**Researched:** 2026-04-24
**Domain:** Jupyter notebook — capstone system integration notebook (NB-12) composing DiT + VAE + T5 + CLIP into the full WanVideoPipeline
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Single large ASCII diagram showing the full pipeline end-to-end — from raw inputs (video, text prompt, reference image) through T5/CLIP encoding, VAE latent encoding, 48-channel concatenation, DiT denoising, and VAE decoding to output video. Tensor shapes annotated at each stage.

**D-02:** Each component box in the diagram includes notebook back-references (e.g., "VAE Encoder (NB-10)", "DiT (NB-06–NB-08)") so readers know where to go for deeper understanding. The diagram acts as a roadmap linking the capstone to all prior notebooks.

**D-03:** Walkthrough with scheduler — show the FlowMatchScheduler timestep setup, walk through one denoising step conceptually (noise prediction → scheduler step → updated latents), and note that production uses 50 steps. Explain flow matching velocity prediction at a level accessible to someone encountering it for the first time. Cover classifier-free guidance (positive vs negative predictions). Do NOT go into full flow matching theory or ELBO math, but explain enough that a reader can follow the denoising code.

**D-04:** Component-by-component demos — create minimal dummy instances of each component (T5 text encoder, CLIP image encoder, VAE, DiT) separately, run each with dummy input, show its output shape. Then show the orchestration (how outputs connect) conceptually via the ASCII diagram. Keeps each cell fast and focused under STD-03's 5-second limit.

**D-05:** Live parameter count computation — instantiate each component and compute `sum(p.numel() for p in model.parameters())`. Produce a breakdown table showing each component's contribution (DiT, VAE, T5, CLIP) with percentages. Consistent with NB-06's LoRA parameter analysis approach.

### Claude's Discretion

- Import/setup strategy (same pattern as Phase 1/2/3 — Claude established it)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within the 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- Whether to use reduced dimensions for text encoder and CLIP (like Phase 2 used reduced block count for DiT)
- How much of the WanVideoPipeline unit architecture to expose — the 20+ unit classes (PromptEmbedder, ImageEmbedderCLIP, FunControl, etc.) are implementation detail; Claude decides how much to surface
- Notebook narrative arc — top-down (pipeline overview first, zoom into components) vs bottom-up (recap, then compose)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SYS-01 | NB-12 covers full pipeline overview — how DiT + VAE + text encoder + CLIP compose into WanVideoPipeline | WanVideoPipeline source fully read; unit architecture documented; composition pattern confirmed |
| SYS-02 | NB-12 shows data flow diagram — from raw inputs through encoding, denoising, decoding to output video | Full ASCII diagram available in model_architecture.md; tensor shapes verified; pipeline source traced end-to-end |
| SYS-03 | NB-12 includes parameter count summary for entire model (~780M DiT params + VAE + encoders) | All component param counts live-computed and verified; critical discrepancy with model_architecture.md found and resolved |
</phase_requirements>

---

## Summary

Phase 4 creates NB-12, the capstone notebook that converges the DiT track (NB-01–NB-08) and VAE track (NB-09–NB-11) into the complete WanVideoPipeline. The notebook shows how all four model components compose, traces any tensor from raw input to output video, and provides a live parameter count table.

The domain is entirely within the existing codebase — no new libraries or external tools are needed. Every component (DiT, VAE, T5, CLIP, FlowMatchScheduler) is already used in prior notebooks. NB-12's job is synthesis and explanation, not new implementation.

**Critical finding:** The `model_architecture.md` file states "~780M DiT params" for the DiT block header, but live computation shows 1.56B for the Fun-Control variant (in_dim=48) and 1.42B for the base T2V variant (in_dim=16). The discrepancy is explained: the blocks alone account for 1.53B (30 × 51.2M per block), and the 780M figure appears nowhere in a reliable reference. NB-12's parameter count cell must use the live-computed values, which contradict model_architecture.md. This is the most important correction this research surfaces.

**Primary recommendation:** NB-12 should follow the top-down narrative arc (pipeline overview first via D-01 diagram, then component demos, then denoising loop, then parameter count). Reduced configs for T5 (vocab=1000, num_layers=2) enable fast CPU demos; production counts are extrapolated using the same per-block-times-N scaling pattern established in NB-06.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Text conditioning | API / Backend | — | T5 encoder runs on CPU/GPU; produces context tensor consumed by DiT cross-attention |
| Image conditioning (CLIP) | API / Backend | — | CLIP ViT-H/14 produces 257-token clip_feature; consumed by DiT img_emb MLP |
| Video latent encoding | API / Backend | — | VAE Encoder3d produces 16-channel latents; feeds both noise initializer and conditioning path |
| Denoising loop | API / Backend | — | FlowMatchScheduler + DiT iterate for 50 steps; entirely in model layer |
| Video latent decoding | API / Backend | — | VAE Decoder3d reconstructs video from final latents |
| Pipeline orchestration | Pipeline Layer | — | WanVideoPipeline units sequence all of the above |
| Notebook presentation | Course / Docs | — | NB-12 provides the explanation layer only; does not modify any source |

---

## Standard Stack

### Core (all already in the project — no new installs)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x (installed) | All tensor operations, model instantiation | Core framework |
| einops | installed | `rearrange` for tensor reshaping | Used across NB-01–NB-11 |
| diffsynth.models.wan_video_dit | local | WanModel, DiTBlock, sinusoidal_embedding_1d | NB-08 reference |
| diffsynth.models.wan_video_vae | local | WanVideoVAE (Encoder3d + Decoder3d) | NB-10/11 reference |
| diffsynth.models.wan_video_text_encoder | local | WanTextEncoder | New to capstone |
| diffsynth.models.wan_video_image_encoder | local | WanImageEncoder | New to capstone |
| diffsynth.diffusion.flow_match | local | FlowMatchScheduler | New to capstone |
| diffsynth.pipelines.wan_video | local | WanVideoPipeline, model_fn_wan_video | New to capstone |

No new pip installs required. [VERIFIED: project source files read and confirmed]

---

## Architecture Patterns

### System Architecture Diagram

```
Raw Inputs
    │
    ├── Text Prompt (str)
    │       │
    │       ▼
    │   WanTextEncoder (T5/UMT5-XXL, 24 layers)         [NB-01 RMSNorm, NB-02 Attention]
    │   vocab: 256,384 → dim: 4,096
    │   output: context [B, L≤512, 4096]
    │
    ├── Reference Image (PIL.Image)
    │       │
    │       ├──▶ WanImageEncoder (CLIP ViT-H/14, 32 layers)  [NB-02, NB-04]
    │       │    224×224 → 257 tokens × 1,280 dim
    │       │    output: clip_feature [B, 257, 1280]
    │       │
    │       └──▶ WanVideoVAE.encode (for reference latent)   [NB-10]
    │            [3,H,W] → [16, 1, H/8, W/8]
    │            output: ref_latents [B, 16, 1, H/8, W/8]
    │
    ├── Control Video (list[PIL.Image])
    │       │
    │       └──▶ WanVideoVAE.encode                          [NB-10]
    │            [3,F,H,W] → [16, T_lat, H/8, W/8]
    │            output: control_latents [B, 16, T_lat, H/8, W/8]
    │
    └── (Noise) Random Normal
            [B, 16, T_lat, H/8, W/8]
            output: noise_latents [B, 16, T_lat, H/8, W/8]

                    ┌────────────────────────────────────────────┐
                    │   48-Channel Concatenation                 │
                    │   cat(noise[16], control[16], y_ref[16])   │
                    │   → x [B, 48, T_lat, H/8, W/8]            │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │   WanModel (DiT, 30 blocks)                 │  [NB-06 blocks,
                    │   patch_embedding: Conv3d(48→1536, k=1,2,2) │   NB-07 patchify,
                    │   Patchify → [B, T*H/2*W/2, 1536] tokens   │   NB-08 forward]
                    │                                             │
                    │   Inputs also injected:                     │
                    │   ├── context [B, L, 1536]    (via text_emb)|
                    │   ├── clip_feature → [B, 257, 1536]         │
                    │   │   prepended to context                  │
                    │   ├── t_mod [B, 6, 1536]      (adaLN)      │
                    │   └── freqs (3D RoPE)          [NB-03]     │
                    │                                             │
                    │   × 50 timesteps via FlowMatchScheduler    │
                    │   (velocity prediction + scheduler.step)   │
                    └──────────────────┬─────────────────────────┘
                                       │
                               denoised latents
                               [B, 16, T_lat, H/8, W/8]
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │   WanVideoVAE.decode                        │  [NB-11]
                    │   Decoder3d upsamples ×8 spatial, ×4 temp  │
                    └──────────────────┬─────────────────────────┘
                                       │
                              Output Video
                              [B, 3, F, H, W]
```

### Recommended Notebook Structure (NB-12)

```
Course/
└── NB-12-pipeline-system-integration.ipynb
    ├── Cell 0: Title, Learning Objectives, Prerequisites
    ├── Cell 1: Setup (same import pattern as NB-01–NB-11)
    ├── Cell 2 (md): Section 1 — Pipeline Overview (D-01, D-02)
    ├── Cell 3: ASCII diagram (as markdown source code block)
    ├── Cell 4 (md): Section 2 — Component Demos (D-04)
    ├── Cell 5: T5 Text Encoder demo (reduced: vocab=1000, num_layers=2)
    ├── Cell 6: CLIP Image Encoder demo
    ├── Cell 7: VAE Encode demo (tiny input)
    ├── Cell 8: DiT forward demo (num_layers=3)
    ├── Cell 9: VAE Decode demo
    ├── Cell 10 (md): How outputs connect (references diagram)
    ├── Cell 11 (md): Section 3 — 48-Channel Composition (SYS-02)
    ├── Cell 12: 48-ch concat code + shape assert
    ├── Cell 13 (md): Section 4 — Denoising Loop (D-03)
    ├── Cell 14: FlowMatchScheduler timesteps demo
    ├── Cell 15: One conceptual denoising step with DiT (3 steps)
    ├── Cell 16 (md): CFG explanation
    ├── Cell 17 (md): Section 5 — Parameter Count (D-05, SYS-03)
    ├── Cell 18: Live param count for each component (reduced configs)
    ├── Cell 19: Production parameter table (extrapolated + verified)
    ├── Cell 20 (md): Summary + Key Takeaways
    └── Cell 21 (md): Exercises (2–3)
```

### Pattern 1: Component-by-Component Demo (D-04)

Each component is instantiated independently with reduced config, run with dummy input, and output shape is asserted. This is the same pattern NB-08 uses for DiT.

```python
# Source: diffsynth/models/wan_video_text_encoder.py
# Demo config: vocab=1000 (vs 256,384), num_layers=2 (vs 24)
# Production architecture: 24 layers, dim=4096, vocab=256,384
text_encoder = WanTextEncoder(
    vocab=1000,      # reduced from 256,384 for speed
    dim=4096,        # production dimension
    dim_attn=4096,
    dim_ffn=10240,
    num_heads=64,
    num_layers=2,    # reduced from 24 for speed
    num_buckets=32,
)
dummy_ids  = torch.zeros(1, 10, dtype=torch.long)   # [B, L]
dummy_mask = torch.ones(1, 10)                       # [B, L]
context = text_encoder(dummy_ids, dummy_mask)        # → [B, L, 4096]
assert context.shape == (1, 10, 4096)
```

```python
# Source: diffsynth/models/wan_video_image_encoder.py
# Full CLIP ViT-H/14 (no reduced version needed — 2.4s init, 0.45s forward)
image_encoder = WanImageEncoder()
dummy_img = torch.randn(1, 3, 224, 224)  # standard ViT-H/14 input
clip_feature = image_encoder.encode_image([dummy_img])  # → [B, 257, 1280]
assert clip_feature.shape == (1, 257, 1280)
```

```python
# Source: diffsynth/models/wan_video_vae.py
# Full VAE with tiny spatial input
vae = WanVideoVAE()
dummy_video = torch.randn(1, 3, 5, 32, 32)  # [B, C, F, H, W]
latents = vae.encode(dummy_video, device='cpu')  # → [B, 16, 2, 4, 4]
decoded = vae.decode(latents, device='cpu')      # → [B, 3, 5, 32, 32]
assert latents.shape[1] == 16  # z_dim
```

```python
# Source: diffsynth/models/wan_video_dit.py, WanModel.__init__
# Demo: 3 blocks (vs 30 production), full production dims
dit = WanModel(
    dim=1536, in_dim=16, ffn_dim=8960, out_dim=16,
    text_dim=4096, freq_dim=256, eps=1e-6,
    patch_size=(1,2,2), num_heads=12,
    num_layers=3,  # 3 for demo; production = 30
    has_image_input=False,
)
```

### Pattern 2: 48-Channel Concatenation (SYS-02, D-01)

```python
# Source: diffsynth/pipelines/wan_video.py, model_fn_wan_video lines ~1250-1252
# and WanVideoUnit_FunControl (lines ~514-528)

B, C, T, H, W = 1, 16, 4, 8, 8  # latent dimensions (tiny for demo)

noise_latents   = torch.randn(B, 16, T, H, W)   # noisy video
control_latents = torch.randn(B, 16, T, H, W)   # VAE-encoded control video
ref_embedding   = torch.randn(B, 16, T, H, W)   # VAE ref + zero-pad (y tensor)

# The 48-channel input is assembled inside model_fn_wan_video:
#   x = latents               → [B, 16, T, H, W]
#   y = cat(control, ref_emb) → [B, 32, T, H, W]
#   x = cat([x, y], dim=1)   → [B, 48, T, H, W]  (in WanModel.forward)
# Then: patch_embedding Conv3d(48→1536) processes the 48-channel tensor

y = torch.cat([control_latents, ref_embedding], dim=1)  # [B, 32, T, H, W]
x = torch.cat([noise_latents, y], dim=1)                # [B, 48, T, H, W]
assert x.shape[1] == 48  # = in_dim of WanModel
```

### Pattern 3: FlowMatchScheduler Denoising Step (D-03)

```python
# Source: diffsynth/diffusion/flow_match.py, set_timesteps_wan + step

scheduler = FlowMatchScheduler("Wan")
scheduler.set_timesteps(50, denoising_strength=1.0, shift=5.0)
# timesteps: 1000.0 → 92.6 (Wan-style shifted linear)
# sigmas:    1.000  → 0.093

# One denoising step:
timestep = scheduler.timesteps[i]                     # e.g., 909.1
sigma    = scheduler.sigmas[i]                        # e.g., 0.9091
sigma_next = scheduler.sigmas[i+1]                    # e.g., 0.8197

noise_pred = dit(latents, timestep.unsqueeze(0), context, ...)  # velocity prediction

# Flow matching update rule:
# prev_sample = sample + velocity * (sigma_next - sigma_current)
latents = scheduler.step(noise_pred, timestep, latents)
# Internally: prev_sample = sample + model_output * (sigma_ - sigma)
# Since sigma_ < sigma, delta is negative → moves toward clean image
```

### Pattern 4: Parameter Count Table (D-05, SYS-03)

```python
# Source: NB-06 approach (lines 11 of NB-06)
# Live computation with reduced configs, then scale to production

components = {
    "DiT (Fun-Control, 3-block demo)": dit_demo,
    "VAE":              vae,
    "T5 (2-layer demo)": text_encoder_demo,
    "CLIP":             image_encoder,
}
for name, model in components.items():
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:30s}: {n:>15,}")

# Production param table (computed in this research session)
# All values VERIFIED via live instantiation
production_counts = {
    "DiT (Fun-Control, in_dim=48)": 1_564_602_176,
    "VAE":                          126_892_531,
    "T5 Text Encoder (UMT5-XXL)":   5_680_910_336,
    "CLIP Image Encoder (ViT-H/14)": 632_076_801,
}
total = sum(production_counts.values())  # 8,004,481,844
```

### Pattern 5: CFG (Classifier-Free Guidance)

```python
# Source: diffsynth/pipelines/wan_video.py lines 301-309
# Run DiT twice: once with positive prompt (posi), once with negative (nega)
noise_pred_posi = model_fn(..., context=posi_context, ...)
noise_pred_nega = model_fn(..., context=nega_context, ...)

# Combine with guidance scale:
noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
# cfg_scale=1.0 → no guidance (only positive)
# cfg_scale=7.5 → typical strength
```

### Anti-Patterns to Avoid

- **Confusing 48-channel composition with channel-stacking vs. WanModel.forward signature:** The WanModel `forward()` takes `x=[B,16,T,H,W]` and `y=[B,32,T,H,W]` separately; it concatenates them internally. Do NOT demonstrate `dit(x_48ch_pre_cat, ...)` because the model's `forward()` does the concat itself. The notebook must show x=[B,16,...] and y=[B,32,...] separately when calling WanModel.forward.
- **Confusing ref_conv tokens with 48-channel input:** `ref_conv` handles a 16-channel single-frame reference image that is encoded as extra tokens in the sequence (not channel-concatenated). This is separate from the 48-ch channel-cat path.
- **Using model_architecture.md's "~780M DiT params" figure:** This is incorrect. The measured value is 1.56B (Fun-Control) or 1.42B (base T2V). NB-12 must use measured values.
- **Applying STD-03 (5s limit) to T5 full instantiation:** Full `WanTextEncoder()` takes ~35s on CPU due to 5.68B parameters. Always use `vocab=1000, num_layers=2` for the demo cell.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Timestep scheduling | Custom sigma schedule | `FlowMatchScheduler("Wan")` | Handles shift, linear spacing, and boundary conditions |
| Flow matching update step | Custom ODE step | `scheduler.step(pred, t, sample)` | Already implements `sample + pred * (sigma_next - sigma)` |
| Tensor-to-token conversion | Custom patchify | `WanModel.patchify(x)` + `unpatchify` | Handles Conv3d, rearrange, grid size tracking |
| CFG combination | Custom weighting | Directly from pipeline: `nega + scale * (posi - nega)` | Standard pattern already in codebase |

---

## Common Pitfalls

### Pitfall 1: 48-Channel Input Shape Mismatch

**What goes wrong:** Passing `x=[B,48,T,H,W]` directly into `WanModel.forward()` instead of `x=[B,16,T,H,W]` + `y=[B,32,T,H,W]` separately.

**Why it happens:** The model's `in_dim=48` suggests passing a 48-channel tensor, but the `forward()` method does `x = torch.cat([x, y], dim=1)` internally when `has_image_input=True` and `require_vae_embedding=True`.

**How to avoid:** Always check `WanModel.forward()` signature at `wan_video_dit.py:359`. For base T2V (no image input), pass just `x=[B,16,T,H,W]`. For Fun-Control, pass `x=[B,16,T,H,W]` and `y=[B,32,T,H,W]`.

**Warning signs:** RuntimeError about channel mismatch in Conv3d (e.g., "expected input to have 48 channels, but got 80 channels").

### Pitfall 2: T5 Instantiation Time Violation

**What goes wrong:** `WanTextEncoder()` (default config) takes ~35 seconds to instantiate due to the 1.05B-param vocabulary embedding (`Embedding(256384, 4096)`).

**Why it happens:** The full vocabulary embedding has 256,384 × 4,096 = 1.05B floating point values that must be initialized at construction time.

**How to avoid:** Use `WanTextEncoder(vocab=1000, num_layers=2)` for demo cells. Show production param count separately as a hardcoded table with a comment explaining the source.

**Warning signs:** Cell taking 30+ seconds on CPU when it should execute under 5s.

### Pitfall 3: Confusing VAE Temporal Compression

**What goes wrong:** Incorrect assumption about the temporal dimension of latents.

**Why it happens:** VAE temporal compression is `(F-1)//4 + 1`, not `F//4`. For 81 frames: `(81-1)//4 + 1 = 21`, not `81//4 = 20`.

**How to avoid:** Always use `(F-1)//4 + 1` formula, not `F//4`. Verify: 5 frames → 2 latent frames (confirmed by live VAE encode test).

**Warning signs:** Shape assertion failures when tracing video → latents → tokens.

### Pitfall 4: model_architecture.md "~780M DiT params" is Wrong

**What goes wrong:** Using 780M as the DiT parameter count in SYS-03 parameter table.

**Why it happens:** `model_architecture.md` states "30 Blocks, ~780M params" but the live measured value is 1.56B (Fun-Control) and 1.42B (base T2V).

**How to avoid:** Compute live via `sum(p.numel() for p in model.parameters())` and use measured values. Document the discrepancy with a note in the notebook.

**Warning signs:** The notebook will produce output inconsistent with prior notebook commentary if 780M is used.

### Pitfall 5: Denoising Loop Direction

**What goes wrong:** Confusion about why `scheduler.step(pred, t, sample)` moves toward the clean image.

**Why it happens:** The step formula is `prev = sample + model_output * (sigma_next - sigma_current)`. Since `sigma_next < sigma_current`, the delta is negative — this is counterintuitive.

**How to avoid:** Explain: flow matching predicts the velocity (direction from noise to data). At t=1000 (sigma=1.0 = pure noise) → t=0 (sigma=0 = clean image), each step adds `velocity * negative_delta_sigma`, which points the sample toward the clean manifold.

---

## Code Examples

### Verified FlowMatchScheduler Usage

```python
# Source: diffsynth/diffusion/flow_match.py, set_timesteps_wan
# Wan scheduler: linear sigmas with shift=5

scheduler = FlowMatchScheduler("Wan")
scheduler.set_timesteps(
    num_inference_steps=50,
    denoising_strength=1.0,  # 1.0 = full denoising from noise
    shift=5.0,               # Wan default; redistributes time budget
)
# scheduler.sigmas:    [1.000, 0.996, ..., 0.093]   (50 values)
# scheduler.timesteps: [1000.0, 995.5, ..., 92.6]   (50 values, = sigmas * 1000)

# One step:
latents = scheduler.step(noise_pred, scheduler.timesteps[i], latents)
# Internally: prev_sample = sample + model_output * (sigma_next - sigma)
```

### Verified add_noise (for video-to-video context)

```python
# Source: diffsynth/diffusion/flow_match.py, add_noise
# Flow matching noising: sample = (1 - sigma) * original + sigma * noise
noisy = scheduler.add_noise(original_latents, noise, timestep=scheduler.timesteps[0])
```

### Verified Pipeline Minimal Denoising Loop

```python
# Source: diffsynth/pipelines/wan_video.py, lines 290-314
scheduler.set_timesteps(3)  # demo with 3 steps
latents = torch.randn(1, 16, T_lat, H_lat, W_lat)

for i, timestep in enumerate(scheduler.timesteps):
    t = timestep.unsqueeze(0)
    # Positive prediction
    noise_pred_posi = dit(latents, t, context)
    # Negative prediction (empty prompt)
    noise_pred_nega = dit(latents, t, context_neg)
    # CFG combination
    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
    # Update latents
    latents = scheduler.step(noise_pred, timestep, latents)
```

---

## Runtime State Inventory

Step 2.5 SKIPPED — this is a greenfield notebook creation phase. No rename/refactor/migration involved.

---

## Environment Availability

All required components are local source files in the `diffsynth/` package.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch | All model ops | ✓ | 2.x | — |
| einops | rearrange calls | ✓ | installed | — |
| WanModel | DiT forward demo | ✓ | local source | — |
| WanVideoVAE | Encode/decode demo | ✓ | local source | — |
| WanTextEncoder | T5 demo | ✓ | local source | — |
| WanImageEncoder | CLIP demo | ✓ | local source | — |
| FlowMatchScheduler | Denoising loop | ✓ | local source | — |
| WanVideoPipeline | Architecture discussion | ✓ | local source | Read-only reference |

**Missing dependencies with no fallback:** None.

**Environment note:** The `diffsynth/__init__.py` triggers a pandas import that fails on this machine (numpy 1.x / 2.x conflict). All prior notebooks use the same `sys.path` workaround — import individual module files directly via `importlib.util` rather than `from diffsynth import ...`. This pattern is already established and must be continued in NB-12. [VERIFIED: observed in NB-08 cell 1]

---

## Validation Architecture

`nyquist_validation` is `false` in `.planning/config.json` — this section is skipped.

---

## Security Domain

No authentication, data persistence, or user-facing API involved. This phase creates a read-only educational notebook. Security section skipped.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DDPM/DDIM scheduler | Flow matching (velocity prediction) | Wan2.1 | DiT predicts velocity (v = noise - data), not noise directly |
| 16-ch T2V DiT input | 48-ch Fun-Control input | Fun-Control model | Enables simultaneous control video + reference image conditioning |
| Standard transformer patchify | Conv3d(in_dim, dim, patch_size) | Wan architecture | Unified spatial+temporal tokenization in one layer |

---

## Verified Parameter Counts (CRITICAL — corrects model_architecture.md)

All values computed via `sum(p.numel() for p in model.parameters())` on live instantiated models. [VERIFIED: live computation, 2026-04-24]

| Component | Config | Measured Params | Notes |
|-----------|--------|----------------|-------|
| DiT (Fun-Control) | dim=1536, in_dim=48, layers=30, has_image_input=True, has_ref_conv=True | **1,564,602,176** (1.56B) | Production config for this repo |
| DiT (Base T2V) | dim=1536, in_dim=16, layers=30, has_image_input=False | **1,418,996,800** (1.42B) | Standard Wan2.1-T2V |
| VAE | WanVideoVAE() defaults | **126,892,531** (127M) | Encoder3d + Decoder3d |
| T5 Text Encoder | vocab=256384, dim=4096, layers=24 | **5,680,910,336** (5.68B) | UMT5-XXL |
| CLIP Image Encoder | WanImageEncoder() defaults (ViT-H/14) | **632,076,801** (632M) | OpenCLIP XLM-Roberta-L |
| **TOTAL (Fun-Control)** | | **8,004,481,844** (**8.00B**) | |

**DiT per-block count:** 51,163,904 params/block × 30 blocks = 1,534,917,120 block params + 29.7M other modules.

**The "~780M DiT" discrepancy:** `model_architecture.md` says "~780M params" for the DiT section header. This is NOT the measured value. The blocks-only count is 1.53B. The full Fun-Control DiT is 1.56B. The "1.3B" product name from Wan-AI likely refers to DiT + VAE + CLIP without T5 (~2.32B, rounded). NB-12's live computation cell will naturally produce the correct measured values.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The "~780M DiT params" figure in model_architecture.md is an error; actual measured is 1.56B | Verified Parameter Counts | LOW — live computation confirms 1.56B; the discrepancy is verified |
| A2 | The notebook narrative arc should be top-down (pipeline first, then components) rather than bottom-up (recap, then compose) | Architecture Patterns | LOW — CONTEXT.md explicitly places this in Claude's Discretion |
| A3 | CLIP `encode_image([img])` takes a list, not a single tensor | Code Examples | LOW — confirmed by reading WanImageEncoder source and WanVideoUnit_ImageEmbedderCLIP usage |

---

## Open Questions

1. **CLIP full instantiation timing (2.4s) + forward (0.45s) = ~2.9s total — borderline STD-03**
   - What we know: CLIP ViT-H/14 with 632M params takes 2.4s to instantiate plus 0.45s for one forward pass = ~2.9s per demo call
   - What's unclear: Whether the cell timing budget is per-call or cumulative across the cell execution
   - Recommendation: Keep CLIP demo isolated in its own cell; should be fine since STD-03 says "execute on CPU in under 5 seconds" and 2.9s < 5s

2. **Scope of WanVideoPipeline unit architecture to expose in NB-12**
   - What we know: There are 20+ unit classes (ShapeChecker, NoiseInitializer, PromptEmbedder, FunControl, FunReference, etc.) — all implementation detail
   - What's unclear: How much to surface in the notebook vs. keeping the explanation at the level of "components orchestrated by units"
   - Recommendation (Claude's Discretion): Show the 5 key units for the Fun-Control flow in prose (PromptEmbedder, FunControl, FunReference, ImageEmbedderCLIP, NoiseInitializer) but do not demo them as runnable code — instead point readers to `wan_video.py` lines 55-82

---

## Sources

### Primary (HIGH confidence)

- `diffsynth/pipelines/wan_video.py` (lines 32–334) — WanVideoPipeline composition, `__call__` denoising loop, `model_fn_wan_video` — read in full
- `diffsynth/models/wan_video_dit.py` (lines 273–416) — WanModel `__init__` and `forward` — read in full
- `diffsynth/diffusion/flow_match.py` — FlowMatchScheduler all methods — read in full
- `model_architecture.md` — Full pipeline diagram and tensor shapes (with noted correction on 780M figure) — read in full
- `.planning/phases/04-system-integration/04-CONTEXT.md` — Locked decisions D-01 through D-05 — read in full
- Live Python computation — all parameter counts, timing measurements, and forward pass demos run 2026-04-24 — VERIFIED

### Secondary (MEDIUM confidence)

- `diffsynth/models/wan_video_text_encoder.py` — WanTextEncoder `__init__` signature and timing characteristics — confirmed via live instantiation
- `diffsynth/models/wan_video_image_encoder.py` — WanImageEncoder timing and output shape — confirmed via live instantiation
- `diffsynth/models/wan_video_vae.py` — WanVideoVAE param count and encode/decode timing — confirmed via live instantiation
- `Course/NB-08-wanmodel-forward.ipynb` — Template pattern, import strategy, cell structure — read cell listing

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are local, already in use, import patterns confirmed
- Architecture: HIGH — code read directly, tensor shapes traced end-to-end, forward passes executed
- Pitfalls: HIGH — discovered through actual execution (T5 timing, 48-ch shape mismatch)
- Parameter counts: HIGH — all values computed via live instantiation, discrepancy in model_architecture.md verified

**Research date:** 2026-04-24
**Valid until:** No external dependencies — valid indefinitely (codebase is read-only)
