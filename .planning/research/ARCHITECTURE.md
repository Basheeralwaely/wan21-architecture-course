# Architecture Patterns: ML Tutorial Notebook Series

**Domain:** Architecture tutorial notebook series — teaching a video diffusion model (Wan2.1) bottom-up
**Researched:** 2026-04-24
**Confidence:** HIGH (based on direct codebase inspection + verified pedagogical patterns from UvA DL, Karpathy Zero-to-Hero, Harvard Annotated Transformer)

---

## Recommended Architecture

A bottom-up series of self-contained Jupyter notebooks in `Course/`, each covering one bounded component.
Each notebook can be read and run independently, but conceptual dependencies flow strictly upward.
No notebook introduces a concept it has not yet earned through prior notebooks or inline explanation.

### Guiding Pedagogical Pattern

This is the **Annotated Walkthrough** pattern, combining:

- **Karpathy-style:** build/run real code in the notebook; mistakes are instructive
- **Harvard Annotated Transformer-style:** code and prose are interleaved; every line is justified
- **UvA DL-style:** each notebook is self-contained and independently runnable; dummy tensors substitute for heavy weights

The reader-contract is: "You know PyTorch tensors, `nn.Module`, and `forward()`. You do not yet know DiT, attention math, or VAE latent compression. This series explains the Wan2.1 source as it actually exists."

---

## Component Boundaries

Each notebook owns a bounded slice of the architecture. Boundaries follow the actual class hierarchy in `diffsynth/models/wan_video_dit.py` and `diffsynth/models/wan_video_vae.py`.

| Notebook | Component | Source file(s) | What the reader learns |
|----------|-----------|---------------|------------------------|
| NB-01 | Primitives: RMSNorm, sinusoidal embedding, modulate | `wan_video_dit.py` lines 64–113 | Layer normalization variants; why RMSNorm over LayerNorm; sinusoidal timestep encoding |
| NB-02 | Linear QKV projections and attention math | `wan_video_dit.py` lines 28–61, 125–148 | What Q, K, V matrices are; dot-product attention; scaled softmax; output projection `o` |
| NB-03 | 3D RoPE (Rotary Positional Encoding) | `wan_video_dit.py` lines 75–98 | 1D vs 3D RoPE; complex-rotation view; why video needs frame/height/width axes |
| NB-04 | SelfAttention + CrossAttention full modules | `wan_video_dit.py` lines 125–187 | Full self-attention with RoPE applied; cross-attention where Q comes from latents, K/V from text context; dual-stream image conditioning |
| NB-05 | Adaptive layer norm conditioning (t_mod, modulate) | `wan_video_dit.py` lines 64–65, 197–232 | How timestep signal reaches each block via `shift/scale/gate`; adaLN-Zero initialisation |
| NB-06 | Single DiT Block | `wan_video_dit.py` lines 197–231 | Self-attention → cross-attention → FFN pipeline; gating residuals; how t_mod flows |
| NB-07 | Patch embedding and unpatchify | `wan_video_dit.py` lines 307–357 | Conv3d patchification of video latents; sequence reshaping `(B,C,F,H,W) → (B,FHW,C)`; inverse |
| NB-08 | Full WanModel (30 DiT blocks end-to-end) | `wan_video_dit.py` lines 273–416 | Input encoding (text, CLIP, timestep); block stack; output head; reference latent prepend |
| NB-09 | VAE primitives: CausalConv3d, ResidualBlock, AttentionBlock | `wan_video_vae.py` lines 33–342 | Causal temporal padding; 3D residual path; single-head spatial attention in the VAE mid-block |
| NB-10 | VAE Encoder: downsampling hierarchy | `wan_video_vae.py` (Encoder class) | 4-level spatial+temporal downsampling; μ/σ split; reparameterisation trick |
| NB-11 | VAE Decoder: upsampling hierarchy | `wan_video_vae.py` (Decoder class) | Mirror of encoder; upsample3d/upsample2d stages; how latent space decodes to pixel frames |
| NB-12 | System integration: full pipeline data flow | `model_architecture.md` + pipeline | How VAE latents feed DiT; what the 48-channel concatenated input means; timestep denoising loop sketch |

---

## Data Flow (Conceptual Dependencies)

Concepts must exist before they can be composed. The dependency chain flows strictly in this direction:

```
Tensor shapes & nn.Module conventions (assumed prior knowledge)
        │
        ▼
NB-01: RMSNorm, sinusoidal_embedding_1d, modulate()
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
NB-02: QKV linear projections         NB-09: CausalConv3d, ResidualBlock
       dot-product attention math             VAE-specific norm + activation
        │                                  │
        ▼                                  ▼
NB-03: 3D RoPE                        NB-10: VAE Encoder (uses NB-09)
  (uses sinusoidal precompute              │
   from NB-01)                            ▼
        │                             NB-11: VAE Decoder (mirror of NB-10)
        ▼                                  │
NB-04: SelfAttention + CrossAttention      │
  (uses QKV from NB-02 + RoPE NB-03)      │
        │                                  │
        ▼                                  │
NB-05: adaLN conditioning (t_mod)          │
  (uses modulate from NB-01)               │
        │                                  │
        ▼                                  │
NB-06: DiT Block                           │
  (uses NB-04 + NB-05)                     │
        │                                  │
        ▼                                  │
NB-07: Patch embedding / unpatchify        │
        │                                  │
        ▼                                  │
NB-08: Full WanModel                       │
  (uses NB-06 + NB-07 + text/CLIP embeds) │
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              NB-12: System integration
              (DiT + VAE + pipeline overview)
```

The VAE path (NB-09 → NB-11) is parallel to the DiT path and does not depend on DiT components. Both paths converge at NB-12.

---

## Suggested Build Order

Build order follows the dependency graph. Numbers match the notebook table above.

**Tier 1 — Atomic building blocks (no prerequisites)**
- NB-01: Norm layers and scalar conditioning utilities
- NB-09: VAE convolution and residual primitives

**Tier 2 — Composed modules (require Tier 1)**
- NB-02: QKV attention (requires NB-01 norm intuition)
- NB-03: 3D RoPE (requires NB-01 sinusoidal embedding)
- NB-10: VAE Encoder (requires NB-09)

**Tier 3 — Integrated sub-systems (require Tier 2)**
- NB-04: Full attention modules (requires NB-02 + NB-03)
- NB-05: Adaptive layer norm conditioning (requires NB-01 modulate)
- NB-11: VAE Decoder (requires NB-10)

**Tier 4 — Block-level assembly (require Tier 3)**
- NB-06: DiT Block (requires NB-04 + NB-05)
- NB-07: Patch embedding (standalone; light dependency on tensor reshaping only)

**Tier 5 — Full models (require Tier 4)**
- NB-08: WanModel end-to-end (requires NB-06 + NB-07)

**Tier 6 — System integration (requires Tier 5)**
- NB-12: Full pipeline data flow (requires NB-08 + NB-11)

This produces the linear notebook sequence: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12.

Readers can jump into the VAE track (09–11) independently after NB-01 if they are only interested in the autoencoder.

---

## Patterns to Follow

### Pattern 1: Dummy tensor scaffolding

Every notebook creates a minimal dummy tensor at the top that matches real shapes. The reader can run every cell without loading model weights.

```python
# Example from NB-04 (SelfAttention)
import torch
from diffsynth.models.wan_video_dit import SelfAttention

B, S, D, N_HEADS = 1, 64, 1536, 12
x = torch.randn(B, S, D)               # seq of latent tokens
freqs = ...                             # precomputed from NB-03

attn = SelfAttention(dim=D, num_heads=N_HEADS)
out = attn(x, freqs)
print(out.shape)  # expect (1, 64, 1536)
```

The dummy shape must exactly match the production shape documented in `model_architecture.md` — e.g. `[B, 22528, 1536]` for the full DiT input sequence. This prevents readers building a wrong mental model.

### Pattern 2: Annotate the source line, then run it

Each cell pair: first cell shows the actual source code excerpt with inline comments; second cell instantiates and runs it with dummy tensors to show output shape. Keeps code walkthrough and execution tightly coupled.

### Pattern 3: Shape tracking as the narrative spine

Print tensor shapes at every stage. Shape transformations _are_ the data flow story. For example, in NB-07 patchify:

```
Input:  (B=1, C=48, F=21, H=64, W=64)   -- video latents
After Conv3d patch embed:  (1, 1536, 21, 32, 32)
After rearrange:  (1, 21504, 1536)       -- flattened sequence
After ref prepend:  (1, 22528, 1536)     -- +1024 ref tokens
```

### Pattern 4: One concept per notebook, one notebook per concept

NB-02 does not explain RoPE. NB-03 does not explain QKV. Keep boundaries clean. Cross-references ("see NB-02 for QKV setup") are fine.

### Pattern 5: Conceptual motivation before code

Every notebook opens with 2–3 markdown cells: what problem this component solves, where it appears in the overall architecture diagram, and what the reader will be able to understand by the end. Then code begins.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Top-down order (pipeline first)

**What happens:** Starting with `WanModel.forward()` or `WanVideoPipeline` before the reader understands attention math forces them to treat sub-modules as black boxes. They can copy-paste but not reason about the model.

**Instead:** Strict bottom-up. NB-08 (WanModel) is earned after NB-01 through NB-07.

### Anti-Pattern 2: Loading real weights

**What happens:** `WanModel.from_pretrained(...)` downloads ~3 GB, requires GPU, and fails in most notebook environments. Real weight values also distract from architectural understanding.

**Instead:** Dummy `torch.randn(...)` tensors sized to match real shapes. Print `.shape` to validate.

### Anti-Pattern 3: Merging too many concepts per notebook

**What happens:** A "DiT Block" notebook that also introduces RoPE and adaLN in the same cells overwhelms the reader and breaks the dependency graph.

**Instead:** Each notebook has exactly one component boundary. NB-06 can _use_ concepts from NB-03 and NB-05 by reference, but does not re-explain them.

### Anti-Pattern 4: Ignoring the 3D nature of video

**What happens:** Explaining attention as if it operates on 2D image patches (height × width). Wan2.1 operates on 3D sequences (frames × height × width) with 3D RoPE. Treating it as 2D teaches the wrong mental model.

**Instead:** NB-03 explicitly shows how `precompute_freqs_cis_3d` splits head dimensions across `(frames, height, width)` axes and why.

### Anti-Pattern 5: Hiding the concatenation trick

**What happens:** Readers of NB-08 are confused why the DiT input has 48 channels (`3 × 16`) rather than 16. If the control latent concatenation is not explained explicitly, the model forward pass is opaque.

**Instead:** NB-12 (or the intro of NB-08) explicitly documents: `input = cat(noise_latents, control_latents, ref_latents, dim=1)` → 48 channels, and why this is how multi-modal conditioning is injected in this model.

---

## Scalability Considerations (Notebook Series)

| Concern | Small series (12 notebooks) | Extension path |
|---------|----------------------------|----------------|
| Reader navigation | Linear numbering (01–12) is sufficient | Add `Course/README.md` with dependency graph if series grows beyond 15 |
| Execution time | All notebooks run in <30s on CPU with dummy tensors | If live attention computation is too slow, further reduce sequence length in dummy tensor |
| Concept reuse | NB-04 references NB-02; markdown cross-links are enough | If concepts drift, extract shared utility cells into `Course/utils.py` importable from notebooks |
| VAE cache | VAE uses `feat_cache` for streaming decode; this is complex for a tutorial | NB-09 should demonstrate `feat_cache=None` path only; cache path flagged as "inference optimization, out of scope" |

---

## Sources

- Direct codebase inspection: `diffsynth/models/wan_video_dit.py`, `diffsynth/models/wan_video_vae.py`, `model_architecture.md`
- UvA Deep Learning Tutorials structure: https://uvadlc-notebooks.readthedocs.io/ (HIGH confidence — verified)
- Harvard Annotated Transformer pedagogical pattern: https://nlp.seas.harvard.edu/annotated-transformer/ (HIGH confidence)
- Karpathy Neural Networks Zero to Hero: https://karpathy.ai/zero-to-hero.html (HIGH confidence)
- HuggingFace Annotated Diffusion Model: https://huggingface.co/blog/annotated-diffusion (MEDIUM confidence — same pattern applied to diffusion)
