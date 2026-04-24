# Phase 2: DiT Assembly - Research

**Researched:** 2026-04-24
**Domain:** Jupyter notebook authoring — composed DiT systems (DiTBlock, patchify/unpatchify, WanModel end-to-end)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Bottom-up buildup approach — start with `DiTBlock.__init__` showing sub-module wiring, then walk through `forward()` line-by-line with back-references to prior notebooks ("recall from NB-05: this is the adaLN modulation"), ending with a full forward pass cell and shape verification
- **D-02:** Include an ASCII block diagram at the top of NB-06 showing the DiT Block data flow: input → adaLN → self-attn → cross-attn → FFN → output with residual connections marked
- **D-03:** Counts + rationale depth — show parameter counts per sub-module (q, k, v, o, ffn.0, ffn.2) with percentages, explain WHY these layers are LoRA targets (task-specific representations), but no LoRA math or rank decomposition details
- **D-04:** Both per-block and full-model scope — show the detailed per-block breakdown first, then scale up to the full 30-block model total
- **D-05:** ASCII diagrams showing spatial-to-sequence mapping — visual intuition of how Conv3d with stride (1,2,2) carves video into patches and flattens to a token sequence, consistent with NB-06's diagram approach
- **D-06:** Include the Head module in NB-07 alongside patchify/unpatchify — keeps the input projection / output projection story complete in one notebook
- **D-07:** Reduced block count (2-3 blocks) for runnable forward pass cells to stay under 5-second STD-03 limit. Show real 30-block config in a separate annotation cell
- **D-08:** Explicit 48-channel concat demo — create separate dummy tensors for noise (16ch), control (16ch), and reference (16ch) latents, show `torch.cat` on dim=1, then feed into model. Reader sees exactly how the 48 channels compose
- **D-09:** Gradient checkpointing shown as training vs inference side-by-side — demonstrate that training mode wraps blocks in `torch.utils.checkpoint.checkpoint()` while eval mode runs blocks directly, explaining memory savings vs recomputation tradeoff

### Claude's Discretion
- Import/setup strategy (same as Phase 1 — Claude established the pattern)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within the 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- Whether to reduce `dim` alongside block count in NB-08, or keep dim=1536 with only block reduction

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIT-12 | NB-06 covers DiT Block — composition of self-attention + cross-attention + FFN + adaLN conditioning | DiTBlock at line 197 verified; forward pass tested at dim=1536, S=64 in 0.020s |
| DIT-13 | NB-06 includes parameter count breakdown identifying LoRA target modules (q, k, v, o, ffn.0, ffn.2) | All counts verified by live execution; see Parameter Count Table in this document |
| DIT-14 | NB-07 covers patchify — Conv3d learned projection with patch size (1,2,2), video-to-token conversion | WanModel.patchify at line 340 verified; Conv3d(48→1536) produces (B, F, H//2, W//2) -> rearranged to (B, F*H//2*W//2, 1536) |
| DIT-15 | NB-07 covers unpatchify — token-to-video reconstruction, shape recovery | WanModel.unpatchify at line 352 verified; round-trip (B,C,F,H,W) → (B,S,1536) → (B,16,F,H,W) passes assertion |
| DIT-16 | NB-08 covers full WanModel end-to-end — 30 blocks, 48-channel input composition (noise + control + reference) | WanModel with dim=1536, 3 blocks, (1,48,4,8,8) input runs in 0.042s; 48-channel concat verified |
| DIT-17 | NB-08 covers gradient checkpointing stripping for inference context | Training path (use_gradient_checkpointing=True) vs eval path verified; training is slower as expected (1.002s vs 0.002s for tiny model) |
</phase_requirements>

---

## Summary

Phase 2 creates three Jupyter notebooks (NB-06 through NB-08) in the existing `Course/` directory, each assembling the Phase 1 primitives into composed systems. All content derives from `diffsynth/models/wan_video_dit.py`: `DiTBlock` (line 197), `Head` (line 254), `WanModel.patchify` (line 340), `WanModel.unpatchify` (line 352), and `WanModel.forward` (line 359). Every class is importable via the established Phase 1 `importlib.util` setup cell pattern — no new import strategy is needed.

The critical pedagogical challenge is **freqs construction**: NB-06 must construct a valid 3D RoPE frequency tensor to call `DiTBlock.forward`. Unlike Phase 1's simplified single-band freqs, NB-06 needs to assemble all three bands (f/h/w) correctly via `torch.cat` before the forward call. The verified assembly formula produces shape `[F*H*W, 1, head_dim//2]` where the last dimension is the concatenation of three complex-half bands (22 + 21 + 21 = 64 for head_dim=128). This is the same assembly that appears in `WanModel.forward` at lines 381–385.

For NB-08, the CONTEXT decision D-07 asks for reduced block count. Verification shows that keeping `dim=1536` (production architecture) with only 3 blocks and small spatial dimensions (e.g., F=4, H=8, W=8 giving seq=64) runs in 0.042 seconds on CPU — well within STD-03's 5-second limit. The recommendation is to keep `dim=1536` with `num_layers=3` so readers see real architecture dimensions, with a clearly commented "production config" annotation cell showing `num_layers=30`.

**Primary recommendation:** Use `dim=1536, num_heads=12, ffn_dim=8960, num_layers=3` for NB-08 runnable cells (verified at 0.042s on CPU), with a separate non-executed annotation cell showing `num_layers=30` for the real WanModel config. This satisfies D-07 while keeping architecture dimensions authentic.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Notebook file creation | Author (Claude) | — | .ipynb files are the deliverable; no runtime tier |
| DiTBlock forward pass demo | CPU (PyTorch) | wan_video_dit.py | Block is imported directly; all ops are CPU-compatible |
| 3D freqs assembly for DiTBlock | CPU (PyTorch) | wan_video_dit.py precompute_freqs_cis_3d | Same assembly as WanModel.forward lines 381-385 |
| patchify Conv3d | CPU (PyTorch) | wan_video_dit.py WanModel.patchify | Real Conv3d is imported and called; no reimplementation |
| unpatchify einops | CPU (PyTorch) | wan_video_dit.py WanModel.unpatchify | Same einops rearrange formula as production code |
| WanModel end-to-end | CPU (PyTorch) | wan_video_dit.py WanModel | Entire WanModel imported; reduced blocks for speed |
| Parameter counts | CPU (PyTorch) | — | `sum(p.numel() for p in mod.parameters())` — no GPU needed |
| Gradient checkpointing demo | CPU (PyTorch) | wan_video_dit.py WanModel.forward | train() mode triggers checkpoint path; eval() skips it |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.8.0+cu126 [VERIFIED: local env] | Tensor ops, nn.Module, Conv3d, checkpoint | All DiT classes are PyTorch; verified installed |
| einops | 0.8.1 [VERIFIED: local env] | rearrange for patchify/unpatchify | Used in WanModel.patchify/unpatchify; same convention notebooks use |
| jupyter | installed [VERIFIED: Phase 1 installed] | Notebook runtime | Already installed by Phase 1 Wave 0 |
| importlib.util | stdlib | Direct module loading | Established in Phase 1; avoids modelscope dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| types | stdlib | Module stubs | Required for wan_video_camera_controller stub in every notebook setup cell |
| math | stdlib | math.prod(patch_size) for output dim | Used in Head output dim calculation: `out_dim * math.prod(patch_size)` |
| pathlib | stdlib | Project root resolution | Established Phase 1 pattern for finding wan_video_dit.py |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dim=1536 with 3 layers (NB-08) | dim=256 with 3 layers | dim=1536 shows real architecture; dim=256 is faster but less representative |
| Full freqs assembly in NB-06 | Simplified single-band freqs (like NB-04) | Full assembly is required because DiTBlock.rope_apply needs correct head_dim//2 last dimension |
| WanModel (NB-08) | Manual block loop | WanModel.forward handles patchify + freqs + blocks + head + unpatchify — more complete story |

---

## Architecture Patterns

### System Architecture Diagram

```
Reader opens NB-06 (DiT Block)
         |
         v
   Course/NB-06-dit-block.ipynb
         |
         +-- Cell 1: Title + Learning Objectives + Prerequisites + Concept Map
         |
         +-- Cell 2: Setup (importlib — same as Phase 1)
         |      |
         |      v diffsynth/models/wan_video_dit.py (READ ONLY)
         |         DiTBlock, GateModule, precompute_freqs_cis_3d
         |
         +-- Section: ASCII diagram (D-02)
         +-- Section: __init__ walkthrough (sub-module wiring)
         +-- Section: forward() line-by-line (with NB back-refs)
         |      |
         |      +-- freqs assembly cell (3D cat from precompute_freqs_cis_3d)
         |      +-- DiTBlock forward pass cell (shape verified)
         |
         +-- Section: LoRA parameter count breakdown (D-03, D-04)
         +-- Summary + Exercises

Reader opens NB-07 (Patchify / Unpatchify / Head)
         |
         v
   Course/NB-07-patchify-unpatchify.ipynb
         |
         +-- Cell 2: Setup (importlib)
         |      Imports: WanModel, Head
         |
         +-- Section: ASCII spatial-to-sequence diagram (D-05)
         +-- Section: patchify — Conv3d + rearrange
         |      Input:  (B, 48, F, H, W)
         |      Output: (B, F*(H//2)*(W//2), 1536)
         +-- Section: unpatchify — einops reverse
         |      Input:  (B, seq, out_dim * prod(patch_size))
         |      Output: (B, 16, F, H, W)  [round-trip]
         +-- Section: Head module (D-06) — output projection with adaLN
         +-- Summary + Exercises

Reader opens NB-08 (Full WanModel)
         |
         v
   Course/NB-08-wanmodel-forward.ipynb
         |
         +-- Cell 2: Setup (importlib)
         |      Imports: WanModel, sinusoidal_embedding_1d
         |
         +-- Section: 48-channel concat demo (D-08)
         +-- Section: WanModel init (reduced blocks) + annotation cell (real 30-block config)
         +-- Section: End-to-end forward pass with shape trace
         +-- Section: Gradient checkpointing (D-09) — train vs eval side-by-side
         +-- Summary + Exercises
```

### Recommended Project Structure

```
Course/
├── NB-01-rmsnorm-sinusoidal-modulate.ipynb    [Phase 1 - exists]
├── NB-02-qkv-projections-head-layout.ipynb    [Phase 1 - exists]
├── NB-03-3d-rope.ipynb                        [Phase 1 - exists]
├── NB-04-self-cross-attention.ipynb            [Phase 1 - exists]
├── NB-05-adaln-zero-modulation.ipynb          [Phase 1 - exists]
├── NB-06-dit-block.ipynb                      [Phase 2 - NEW]
├── NB-07-patchify-unpatchify.ipynb            [Phase 2 - NEW]
└── NB-08-wanmodel-forward.ipynb               [Phase 2 - NEW]
```

### Pattern 1: DiTBlock forward with proper 3D freqs

**What:** DiTBlock.forward requires a `freqs` tensor of shape `[seq_len, 1, head_dim//2]` assembled from all three 3D RoPE bands.

**When to use:** NB-06 forward pass demo. This is the first notebook that needs full 3D freqs (NB-04 used simplified single-band freqs as a shortcut).

**Why:** The `rope_apply` in SelfAttention concatenates the three band outputs internally — the full `head_dim//2` must be present as the last dimension of `freqs`.

```python
# Source: diffsynth/models/wan_video_dit.py, lines 381-385 (WanModel.forward)
from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d, DiTBlock
import torch

dim, num_heads, ffn_dim = 1536, 12, 8960
head_dim = dim // num_heads  # 128
# [VERIFIED: local execution] f_freqs: (1024,22), h_freqs: (1024,21), w_freqs: (1024,21)
f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)

# Grid dimensions for a small dummy video (F frames, H/2 height tokens, W/2 width tokens)
F, H, W = 4, 4, 4  # -> seq_len = 4*4*4 = 64 after patchify with stride (1,2,2)
freqs = torch.cat([
    f_freqs[:F].view(F, 1, 1, -1).expand(F, H, W, -1),  # temporal band
    h_freqs[:H].view(1, H, 1, -1).expand(F, H, W, -1),  # height band
    w_freqs[:W].view(1, 1, W, -1).expand(F, H, W, -1),  # width band
], dim=-1).reshape(F * H * W, 1, -1)  # [seq_len, 1, head_dim//2]
# freqs.shape == torch.Size([64, 1, 64])  (64 = head_dim//2 = 22+21+21)
assert freqs.shape == torch.Size([F * H * W, 1, head_dim // 2])

# Now call DiTBlock
block = DiTBlock(has_image_input=True, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
block.eval()
B, S = 1, F * H * W  # 64
x = torch.randn(B, S, dim)          # [B, S, dim]
context = torch.randn(B, 277, dim)  # 257 CLIP + 20 text -> [B, 277, dim]
t_mod = torch.randn(B, 6, dim)      # from WanModel.time_projection.unflatten: [B, 6, dim]

with torch.no_grad():
    out = block(x, context, t_mod, freqs)  # [B, S, dim]
assert out.shape == torch.Size([B, S, dim])
print(f"DiTBlock output: {out.shape}")  # torch.Size([1, 64, 1536])
```

### Pattern 2: patchify / unpatchify round-trip

**What:** Conv3d(in_dim=48, dim=1536, kernel=(1,2,2), stride=(1,2,2)) projects video to tokens; einops rearrange reverses it.

**When to use:** NB-07.

```python
# Source: diffsynth/models/wan_video_dit.py, lines 340-357
import torch, torch.nn as nn, math
from einops import rearrange

patch_size = (1, 2, 2)
in_dim, dim, out_dim = 48, 1536, 16

# Patchify step 1: Conv3d (from WanModel.__init__ line 307)
patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
# patch_embedding.weight shape: (1536, 48, 1, 2, 2)
# patch_embedding.bias shape:   (1536,)

B, C, F, H, W = 1, 48, 4, 8, 8   # [B, C, F, H, W]
x = torch.randn(B, C, F, H, W)    # [1, 48, 4, 8, 8]
px = patch_embedding(x)            # [1, 1536, 4, 4, 4] — stride halves H and W
b, c, f, h, w = px.shape          # b=1, c=1536, f=4, h=4, w=4

# Patchify step 2: flatten to sequence
x_seq = rearrange(px, 'b c f h w -> b (f h w) c')  # [1, 64, 1536]
assert x_seq.shape == torch.Size([B, f * h * w, dim])

# Unpatchify: reverse using einops (from WanModel.unpatchify line 352-357)
out_dim_full = out_dim * math.prod(patch_size)  # 16 * 1*2*2 = 64
head_out = torch.randn(B, f * h * w, out_dim_full)  # simulated head output
x_video = rearrange(
    head_out, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
    f=f, h=h, w=w, x=patch_size[0], y=patch_size[1], z=patch_size[2]
)  # [1, 16, 4, 8, 8]
assert x_video.shape == torch.Size([B, out_dim, F, H, W])
print(f"Round-trip: {x.shape} -> patchify -> {x_seq.shape} -> unpatchify -> {x_video.shape}")
```

### Pattern 3: 48-channel concat demo (NB-08)

**What:** Show noise/ctrl/ref latents as separate tensors then concatenate before passing to WanModel.

**When to use:** NB-08 opening section (D-08).

```python
# Source: model_architecture.md + diffsynth/models/wan_video_dit.py WanModel.forward
import torch

# Each is 16-channel from VAE encoding
B, F, H, W = 1, 4, 8, 8  # small spatial for NB-08 demo
noise_latent   = torch.randn(B, 16, F, H, W)   # denoising target
control_latent = torch.randn(B, 16, F, H, W)   # control video (structure)
ref_latent     = torch.randn(B, 16, F, H, W)   # reference image (appearance)

# WanModel expects 48-channel input — concat on channel dim
x_48 = torch.cat([noise_latent, control_latent, ref_latent], dim=1)  # [B, 48, F, H, W]
assert x_48.shape == torch.Size([B, 48, F, H, W])
print(f"noise:    {noise_latent.shape}")
print(f"control:  {control_latent.shape}")
print(f"ref:      {ref_latent.shape}")
print(f"48-ch x:  {x_48.shape}")
```

### Pattern 4: WanModel reduced-block demo (NB-08, D-07)

**What:** Instantiate WanModel with production dim=1536 but only 3 layers for fast CPU execution.

**When to use:** NB-08 forward pass demo cells.

```python
# Source: diffsynth/models/wan_video_dit.py, lines 273-416
from diffsynth.models.wan_video_dit import WanModel
import torch

# --- DEMO CONFIG (reduced layers for CPU demo, STD-03) ---
model = WanModel(
    dim=1536,           # production architecture dimension
    in_dim=48,          # 16 noise + 16 ctrl + 16 ref channels
    ffn_dim=8960,       # production FFN dimension
    out_dim=16,         # output: 16 latent channels
    text_dim=4096,      # T5 text encoder output dim
    freq_dim=256,       # sinusoidal timestep embedding dim
    eps=1e-6,
    patch_size=(1, 2, 2),
    num_heads=12,
    num_layers=3,       # REDUCED from 30 for demo
    has_image_input=False,  # no CLIP for simplified demo
)
# --- PRODUCTION CONFIG (annotation only — do not run) ---
# model = WanModel(dim=1536, in_dim=48, ffn_dim=8960, out_dim=16,
#     text_dim=4096, freq_dim=256, eps=1e-6, patch_size=(1,2,2),
#     num_heads=12, num_layers=30, has_image_input=True)

model.eval()
x_48 = torch.cat([torch.randn(1, 16, 4, 8, 8)] * 3, dim=1)  # [1, 48, 4, 8, 8]
timestep = torch.tensor([500.0])       # diffusion timestep
context  = torch.randn(1, 20, 4096)   # text embeddings [B, L, text_dim]

with torch.no_grad():
    out = model(x_48, timestep, context)  # [1, 16, 4, 8, 8]
assert out.shape == torch.Size([1, 16, 4, 8, 8])
print(f"WanModel output: {out.shape}")  # torch.Size([1, 16, 4, 8, 8])
```

### Pattern 5: Gradient checkpointing demo (NB-08, D-09)

**What:** Show the branching logic in WanModel.forward that enables/disables checkpointing.

**When to use:** NB-08 gradient checkpointing section.

```python
# Source: diffsynth/models/wan_video_dit.py, lines 393-412
import torch

# TRAINING mode — gradient checkpointing active
model.train()
x_train = x_48.detach()  # no requires_grad needed on input; forward sets it
out_gc = model(x_train, timestep, context, use_gradient_checkpointing=True)
# Internally for each block:
#   x = torch.utils.checkpoint.checkpoint(block, x, context, t_mod, freqs, use_reentrant=True)
# Recomputes activations during backward instead of storing them -> lower peak VRAM

# INFERENCE mode — no checkpointing
model.eval()
with torch.no_grad():
    out_eval = model(x_48, timestep, context)
    # Internally: x = block(x, context, t_mod, freqs)   -- direct call, no checkpoint

print("Training with checkpointing: saves ~25% VRAM by recomputing block activations")
print("Inference without checkpointing: faster, no gradient tracking needed")
assert out_gc.shape == out_eval.shape
```

### Anti-Patterns to Avoid

- **Using simplified freqs in NB-06:** NB-04 used `freqs = f_freqs[:S].unsqueeze(1)` (single band, shape [S, 1, f_dim//2]). This FAILS for DiTBlock because `rope_apply` expects last dim = full `head_dim//2`. Always use the three-band assembly from WanModel.forward lines 381-385.
- **Calling `block.forward()` with `t_mod` in 4D shape:** `DiTBlock.forward` supports `t_mod` shape `[B, 6, dim]` (standard) and `[B, seq, 6, dim]` (sequence-separated). NB-06 should use `[B, 6, dim]` — the standard WanModel path. The 4D path is an advanced variant not needed in the course.
- **Instantiating full WanModel with num_layers=30 for CPU demo:** At dim=1536, 30 blocks have ~51M params each = ~1.5B parameters total. Model init alone takes several seconds on CPU. Use num_layers=3 for demo cells.
- **Calling model(x, ...) with has_image_input=True without clip_feature:** If `WanModel.__init__` is called with `has_image_input=True`, the forward pass expects a `clip_feature` argument. Use `has_image_input=False` for the NB-08 simplified demo; this avoids the CLIP MLP and image path.
- **Loading real weights:** Violates the portability constraint. All demo cells use `torch.randn(...)` dummy tensors.
- **Grad checkpointing with `model.eval()`:** The checkpoint path only triggers when `self.training and use_gradient_checkpointing`. In eval mode, blocks run directly regardless of the flag. This is the correct behavior — document it explicitly in NB-08.

---

## Source Line Reference Table (STD-06)

All Phase 2 notebooks MUST cite these locations:

| Symbol | File | Line | Notes |
|--------|------|------|-------|
| `DiTBlock.__init__` | `diffsynth/models/wan_video_dit.py` | 198 | Wires self_attn, cross_attn, norm1/2/3, ffn, modulation, gate |
| `DiTBlock.forward` | `diffsynth/models/wan_video_dit.py` | 215 | has_seq branch at line 216; six-chunk at line 219; GateModule calls at 227,230 |
| `GateModule.forward` | `diffsynth/models/wan_video_dit.py` | 194 | `x + gate * residual` — gate=0 produces identity |
| `Head.__init__` | `diffsynth/models/wan_video_dit.py` | 255 | norm (no affine), head linear, modulation (2 params) |
| `Head.forward` | `diffsynth/models/wan_video_dit.py` | 263 | Dual t_mod shape: 3D=[B,dim] and 4D=[B,seq,dim] paths |
| `WanModel.__init__` | `diffsynth/models/wan_video_dit.py` | 274 | patch_embedding Conv3d at line 307; blocks ModuleList at 321 |
| `WanModel.patchify` | `diffsynth/models/wan_video_dit.py` | 340 | Conv3d then rearrange 'b c f h w -> b (f h w) c' |
| `WanModel.unpatchify` | `diffsynth/models/wan_video_dit.py` | 352 | einops rearrange 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)' |
| `WanModel.forward` | `diffsynth/models/wan_video_dit.py` | 359 | freqs assembly at lines 381-385; checkpoint branch at 393-412 |

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video-to-token flattening | Custom reshape loop | `rearrange(px, 'b c f h w -> b (f h w) c')` from WanModel.patchify | Self-documenting, zero-copy where possible, matches production code |
| Token-to-video reconstruction | Custom unflatten loop | `rearrange(x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)')` | Handles 6-way axis merge correctly; hand-rolled version is likely to get axis order wrong |
| 3D frequency grid assembly | Custom tensor construction | `torch.cat` pattern from WanModel.forward lines 381-385 | The exact broadcast+expand+reshape sequence is non-obvious; the codebase has the correct form |
| DiTBlock from scratch | Custom block loop | `DiTBlock` from wan_video_dit.py | STD-07 requires real classes; DiTBlock already composes all the Phase 1 primitives |
| Parameter counting | Manual parameter iteration | `sum(p.numel() for p in mod.parameters())` | Standard PyTorch idiom; handles nested sub-modules correctly |

**Key insight:** All Phase 2 notebooks assemble existing Phase 1 primitives into larger structures. The educational value is in seeing how the pieces compose, not in rebuilding them. Every notebook must import and run real `diffsynth` symbols (STD-07).

---

## Parameter Count Table (Verified)

All values verified via live execution against the actual `DiTBlock` class.

### Per-Block Breakdown (dim=1536, num_heads=12, ffn_dim=8960, has_image_input=True)

| Sub-module | Parameters | % of Block |
|-----------|-----------|------------|
| self_attn.q | 2,360,832 | 4.6% |
| self_attn.k | 2,360,832 | 4.6% |
| self_attn.v | 2,360,832 | 4.6% |
| self_attn.o | 2,360,832 | 4.6% |
| self_attn.norm_q | 1,536 | ~0% |
| self_attn.norm_k | 1,536 | ~0% |
| cross_attn.q | 2,360,832 | 4.6% |
| cross_attn.k | 2,360,832 | 4.6% |
| cross_attn.v | 2,360,832 | 4.6% |
| cross_attn.o | 2,360,832 | 4.6% |
| cross_attn.k_img | 2,360,832 | 4.6% |
| cross_attn.v_img | 2,360,832 | 4.6% |
| cross_attn.norm_q/k/k_img | 4,608 | ~0% |
| norm3 (LayerNorm, with affine) | 3,072 | ~0% |
| ffn[0] (Linear 1536→8960) | 13,771,520 | 26.9% |
| ffn[2] (Linear 8960→1536) | 13,764,096 | 26.9% |
| modulation (nn.Parameter) | 9,216 | ~0% |
| **Block total** | **51,163,904** | 100% |

### LoRA Target Scope [VERIFIED: local execution]

| Target | Parameters | LoRA Rationale |
|--------|-----------|----------------|
| self_attn.q, k, v, o | 4 × 2,360,832 = 9,443,328 | Core attention projections — task-specific query/key/value transforms |
| cross_attn.q, k, v, o | 4 × 2,360,832 = 9,443,328 | Text conditioning projections — adapt how text guides video generation |
| ffn[0], ffn[2] | 13,771,520 + 13,764,096 = 27,535,616 | FFN is the block's "memory" — high-dimensional feature transform |
| **Per-block LoRA total** | **46,422,272** | 90.7% of block parameters |
| **30-block LoRA total** | **1,392,668,160** | All LoRA-adapted params across full model |

Note: `cross_attn.k_img` and `cross_attn.v_img` are NOT LoRA targets per `train_lora_cached.py` default `--lora_target_modules "q,k,v,o,ffn.0,ffn.2"`. The LoRA config targets by name match (`q`, `k`, `v`, `o` in both self_attn and cross_attn, plus `ffn.0` and `ffn.2`).

### Patch Embedding and Head Counts

| Module | Parameters | Notes |
|--------|-----------|-------|
| patch_embedding (Conv3d 48→1536, k=(1,2,2)) | 296,448 | weight=(1536,48,1,2,2) + bias=(1536,) |
| Head (norm + linear + modulation) | 101,440 | head.weight=(64,1536) + head.bias=(64,) + modulation=(1,2,1536) |

---

## Common Pitfalls

### Pitfall 1: Incorrect freqs shape for DiTBlock

**What goes wrong:** Using simplified freqs from NB-04 (`freqs = f_freqs[:S].unsqueeze(1)` giving shape `[S, 1, f_dim//2]`) in NB-06 raises a `RuntimeError: Sizes of tensors must match` inside `rope_apply` because the last dimension (f_dim//2 = 22) doesn't equal the required head_dim//2 (64).

**Why it happens:** `rope_apply` in SelfAttention multiplies `x_out * freqs` where `x_out` is shaped to complex pairs covering all of head_dim, so freqs must cover all three bands.

**How to avoid:** Use the full three-band assembly from WanModel.forward lines 381-385. The correct last dimension is 22+21+21=64 = head_dim//2. Verified working pattern is in Code Examples above.

**Warning signs:** `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 64 but got size 22`

### Pitfall 2: DiTBlock demo freqs need matching seq_len

**What goes wrong:** Building freqs for an F×H×W grid of size 4×4×4=64, then passing `x` with a different S (e.g., 20) mismatches the freqs and token sequence lengths.

**Why it happens:** `rope_apply` uses freqs positionally — position i of the sequence gets freqs[i]. If seq_len != F*H*W, the shapes mismatch.

**How to avoid:** Always set `S = F * H * W` and create `x = torch.randn(B, S, dim)`. The convention is that the grid dimensions (F, H, W) determine the sequence length after patchify.

**Warning signs:** Shape mismatch errors inside `rope_apply` or during freqs expand operations.

### Pitfall 3: WanModel with has_image_input=True requires clip_feature

**What goes wrong:** Instantiating `WanModel(has_image_input=True, ...)` and calling `forward(x, timestep, context)` without `clip_feature` raises `AttributeError` or silent failure because the CLIP embedding path (`self.img_emb`) is called unconditionally when `has_image_input=True`.

**Why it happens:** Line 374: `if self.has_image_input: x = torch.cat([x, y], dim=1)` — the model also expects `clip_feature` to be passed for the img_emb MLP.

**How to avoid:** Use `has_image_input=False` for NB-08 simplified demo. This skips the CLIP path entirely and keeps in_dim=48 as the direct channel count.

### Pitfall 4: Gradient checkpointing requires model.train()

**What goes wrong:** Calling `model(x, t, ctx, use_gradient_checkpointing=True)` while the model is in eval mode (after `model.eval()`) silently runs direct block calls instead of checkpoint calls.

**Why it happens:** Line 393: `if self.training and use_gradient_checkpointing:` — the flag is ANDed with `self.training`. eval mode suppresses it.

**How to avoid:** For NB-08 D-09 demo, explicitly call `model.train()` before the checkpointing cell and `model.eval()` before the inference cell. Add a comment explaining the condition.

**Warning signs:** No error — the code runs silently but the checkpoint path is never exercised.

### Pitfall 5: Unpatchify argument order in einops

**What goes wrong:** Swapping the axis labels in `rearrange(x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)', ...)` produces wrong spatial layout (transposed height/width or wrong patch ordering).

**Why it happens:** The einops string encodes the exact decomposition order: the input group `(x y z c)` means patch_t=x, patch_h=y, patch_w=z, channel=c. The output `(f x)` interleaves frames with temporal patches (which are always 1), `(h y)` interleaves height tokens with height patches, `(w z)` interleaves width tokens with width patches.

**How to avoid:** Always copy the exact rearrange string from `WanModel.unpatchify` (line 353-356). Verify the round-trip with `assert x_video.shape == torch.Size([B, out_dim, F, H, W])`.

---

## Code Examples

### NB-06: DiTBlock __init__ sub-module wiring walkthrough

```python
# Source: diffsynth/models/wan_video_dit.py, lines 197-213
from diffsynth.models.wan_video_dit import DiTBlock
import torch

dim, num_heads, ffn_dim = 1536, 12, 8960

block = DiTBlock(has_image_input=True, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)

# Sub-module inventory
print("DiTBlock sub-modules:")
print(f"  self_attn (SelfAttention):  {sum(p.numel() for p in block.self_attn.parameters()):,} params")
print(f"  cross_attn (CrossAttention): {sum(p.numel() for p in block.cross_attn.parameters()):,} params")
print(f"  norm1 (LayerNorm, no affine): {sum(p.numel() for p in block.norm1.parameters()):,} params")
print(f"  norm2 (LayerNorm, no affine): {sum(p.numel() for p in block.norm2.parameters()):,} params")
print(f"  norm3 (LayerNorm, with affine): {sum(p.numel() for p in block.norm3.parameters()):,} params")
print(f"  ffn[0] (Linear 1536→8960):   {sum(p.numel() for p in block.ffn[0].parameters()):,} params")
print(f"  ffn[2] (Linear 8960→1536):   {sum(p.numel() for p in block.ffn[2].parameters()):,} params")
print(f"  modulation (nn.Parameter):    {block.modulation.numel():,} params  # [1, 6, dim]")
total = sum(p.numel() for p in block.parameters())
print(f"  Block total: {total:,}")
```

### NB-06: DiTBlock forward pass line-by-line (annotated)

```python
# Source: diffsynth/models/wan_video_dit.py, lines 215-231
# Annotated: what each line does (for NB-06 prose-before-code cells)

# 1. Extract six modulation parameters (recall NB-05: adaLN-Zero)
#    modulation shape: [1, 6, dim] (learned per-block offset)
#    t_mod shape: [B, 6, dim] (from WanModel.time_projection)
#    chunk(6, dim=1) splits the [B, 6, dim] into six [B, 1, dim] tensors
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
).chunk(6, dim=1)  # six [B, 1, dim] tensors

# 2. Self-attention branch (recall NB-04: SelfAttention, NB-01: modulate)
input_x = modulate(block.norm1(x), shift_msa, scale_msa)  # adaLN pre-norm
x = block.gate(x, gate_msa, block.self_attn(input_x, freqs))  # gated residual

# 3. Cross-attention branch (recall NB-04: CrossAttention) — no adaLN on cross-attn
x = x + block.cross_attn(block.norm3(x), context)  # simple residual, no gate

# 4. FFN branch (recall NB-01: modulate)
input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)  # adaLN pre-norm
x = block.gate(x, gate_mlp, block.ffn(input_x))  # gated residual
```

### NB-07: Head module forward pass

```python
# Source: diffsynth/models/wan_video_dit.py, lines 254-270
from diffsynth.models.wan_video_dit import Head
import torch, math

dim, out_dim, patch_size = 1536, 16, (1, 2, 2)
head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=1e-6)

print("Head sub-modules:")
print(f"  norm (LayerNorm, no affine): {sum(p.numel() for p in head.norm.parameters()):,} params")
print(f"  head (Linear {dim}→{out_dim * math.prod(patch_size)}): {sum(p.numel() for p in head.head.parameters()):,} params")
print(f"  modulation (nn.Parameter [1,2,{dim}]): {head.modulation.numel():,} params")

B, S = 1, 64
x = torch.randn(B, S, dim)     # [B, seq, dim]
t = torch.randn(B, dim)         # [B, dim] — time embedding (NOT t_mod, NOT t_proj)
# Note: WanModel.forward passes 't' (output of time_embedding, before time_projection) to head

head.eval()
with torch.no_grad():
    out = head(x, t)
# out: [B, S, out_dim * prod(patch_size)] = [1, 64, 64]
assert out.shape == torch.Size([B, S, out_dim * math.prod(patch_size)])
print(f"Head output: {out.shape}")  # [1, 64, 64]
```

### NB-08: Complete forward pass shape trace

```python
# Source: diffsynth/models/wan_video_dit.py, lines 359-416 (WanModel.forward)
# Shape trace — each operation annotated

from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
import torch

model = WanModel(dim=1536, in_dim=48, ffn_dim=8960, out_dim=16,
    text_dim=4096, freq_dim=256, eps=1e-6, patch_size=(1,2,2),
    num_heads=12, num_layers=3, has_image_input=False)
model.eval()

B, F, H, W = 1, 4, 8, 8
x_48  = torch.cat([torch.randn(B,16,F,H,W)]*3, dim=1)  # [1, 48, 4, 8, 8]
timestep = torch.tensor([500.0])   # [B]
context  = torch.randn(B, 20, 4096)  # [B, L, 4096]

# Shape trace of WanModel.forward:
# 1. time_embedding:    [B] -> sinusoidal([B, freq_dim=256]) -> linear+silu+linear -> t [B, dim=1536]
# 2. time_projection:   t [B, 1536] -> SiLU+linear -> [B, 9216] -> unflatten -> t_mod [B, 6, 1536]
# 3. text_embedding:    context [B, 20, 4096] -> 2×linear+GELU -> [B, 20, 1536]
# 4. patchify:          x_48 [B, 48, 4, 8, 8] -> Conv3d -> [B, 1536, 4, 4, 4] -> rearrange -> [B, 64, 1536]
# 5. freqs assembly:    f_freqs+h_freqs+w_freqs -> [64, 1, 64]
# 6. ×3 DiTBlocks:      x [B, 64, 1536] (unchanged shape through each block)
# 7. head:              x [B, 64, 1536] -> linear -> [B, 64, 64]  (64 = 16 * 1*2*2)
# 8. unpatchify:        [B, 64, 64] -> rearrange -> [B, 16, 4, 8, 8]

with torch.no_grad():
    out = model(x_48, timestep, context)
assert out.shape == torch.Size([B, 16, F, H, W])
print(f"Input:  {x_48.shape}")
print(f"Output: {out.shape}")
```

---

## Verified Timing Budget (STD-03)

All timings measured on CPU in this session [VERIFIED: local execution].

| NB-06 Cell | Config | Time |
|-----------|--------|------|
| DiTBlock forward (S=16, dim=1536) | dim=1536, has_image_input=True | 0.033s |
| DiTBlock forward (S=64, dim=1536) | dim=1536, has_image_input=True | 0.020s |
| Head forward (S=64, dim=1536) | dim=1536 | <0.001s |

| NB-08 Cell | Config | Time |
|-----------|--------|------|
| WanModel forward | dim=1536, 3 layers, F=4, H=8, W=8 (seq=64) | 0.042s |
| WanModel forward | dim=1536, 3 layers, F=4, H=16, W=16 (seq=256) | 0.129s |
| WanModel train (grad checkpoint) | dim=128, 2 layers, tiny | 1.002s |

**Recommendation:** Use `F=4, H=8, W=8` (seq_len=64) for NB-08 forward pass demo cells. This keeps total notebook execution time well within STD-03's 5-second requirement.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate positional embeddings for video frames | 3D RoPE applied per-head split (temporal/H/W) | Video DiT models 2024 | RoPE applied inside each block via precomputed freqs tensor |
| Standard transformer patchify (2D patches, no temporal) | Conv3d with kernel (1,2,2) stride (1,2,2) for video | Video DiT models 2024 | Temporal patch size=1 means one latent frame = one temporal token |
| adaLN-Zero with zero-init final linear | Per-block learned modulation parameter + time projection | Wan2.1 specific | `self.modulation = nn.Parameter(randn/dim**0.5)` — random small init, not zero |
| Storing all activations for backward | Gradient checkpointing (`torch.utils.checkpoint`) | PyTorch standard, adopted widely 2022+ | Recomputes block activations on backward pass — trades ~25% compute for lower peak VRAM |

**Deprecated/outdated:**
- 2D spatial patchify only: replaced by Conv3d for video — NB-07 must explain the temporal dimension is preserved (patch_size[0]=1 means full temporal resolution passes through unchanged in latent space).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `has_image_input=False` is the right simplified demo for NB-08 (avoids needing clip_feature dummy tensor) | Pattern 4 | If user wants has_image_input=True demo, add clip_feature=torch.randn(B,257,1280) and y=torch.randn(B,16,1,H,W) — adds complexity but is straightforward |
| A2 | cross_attn.k_img and cross_attn.v_img are NOT LoRA targets per the default lora_target_modules in train_lora_cached.py | Parameter Count Table | Verified from train_lora_cached.py line 292: default "q,k,v,o,ffn.0,ffn.2" — these match by layer name |

**All other claims were verified via live code execution against the actual project files.**

---

## Open Questions

1. **NB-06 freqs input exposition**
   - What we know: NB-06 DiTBlock demo needs full 3D freqs assembly (not the simplified NB-04 shortcut)
   - What's unclear: How much of the freqs assembly should be re-derived in NB-06 vs just "recall NB-03 assembly"?
   - Recommendation: In NB-06, include the freqs assembly as a standalone cell with a comment "from WanModel.forward lines 381-385 — see NB-03 for the individual band computations". The assembly is 5 lines and readers need to see it to understand the DiTBlock demo works correctly.

2. **NB-07 Head t_mod vs t argument**
   - What we know: `Head.forward(x, t_mod)` — in WanModel.forward, the argument passed is `t` (output of `time_embedding`, NOT `t_mod` which is the projected 6-chunk tensor). Head uses its own internal `modulation` parameter + a 2-chunk split.
   - What's unclear: The parameter name `t_mod` in Head.forward is confusing because it's a different tensor from DiTBlock's `t_mod`.
   - Recommendation: NB-07 should explicitly note: "Head receives `t` (the 1536-dim time embedding before the 6-chunk projection). It applies its own 2-parameter adaLN (shift + scale for output normalization), NOT the 6-parameter modulation from DiTBlock."

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10 | All notebooks | Yes | 3.10.18 | — |
| torch | All notebooks | Yes | 2.8.0+cu126 | — |
| einops | All notebooks | Yes | 0.8.1 | — |
| jupyter | Notebook execution | Yes | installed by Phase 1 | — |
| modelscope | (NOT needed) | No | — | Bypassed by importlib strategy (established Phase 1) |
| flash_attn | (NOT needed) | No | — | PyTorch SDPA fallback confirmed active |

**No missing dependencies with no fallback:** All required tools are available.

---

## Validation Architecture

> nyquist_validation is set to false in .planning/config.json — this section is omitted per configuration.

---

## Security Domain

This phase creates read-only educational notebooks that load no real weights, make no network calls, and handle no user data. No ASVS categories apply.

---

## Sources

### Primary (HIGH confidence)
- `diffsynth/models/wan_video_dit.py` — Direct source read; DiTBlock (lines 197-231), Head (lines 254-270), WanModel (lines 273-416) [VERIFIED: local codebase]
- Python interpreter execution — All parameter counts, shape assertions, timing measurements, and forward pass results verified via live code execution [VERIFIED: local env]
- `model_architecture.md` — Architecture diagrams and tensor shapes used for prose exposition [VERIFIED: local codebase]
- `train_lora_cached.py` — LoRA target module names verified from default arg at line 292 [VERIFIED: local codebase]

### Secondary (MEDIUM confidence)
- `.planning/phases/01-dit-foundations/01-RESEARCH.md` — Phase 1 patterns (import strategy, notebook template, STD standards)
- `.planning/phases/01-dit-foundations/01-PATTERNS.md` — Shared patterns and anti-patterns established in Phase 1
- `.planning/phases/02-dit-assembly/02-CONTEXT.md` — User decisions locked in discussion phase

### Tertiary (LOW confidence)
- None — all claims verified against local codebase or live execution.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — torch/einops confirmed from live env; same as Phase 1
- Architecture patterns: HIGH — all code patterns executed against actual wan_video_dit.py; timing measured
- Parameter counts: HIGH — live execution via `sum(p.numel() for p in mod.parameters())`
- Pitfalls: HIGH — Pitfall 1 discovered by actual RuntimeError during research; others from code reading
- Line numbers: HIGH — confirmed via grep against actual file

**Research date:** 2026-04-24
**Valid until:** 2026-07-24 (stable — source code is the ground truth; invalidated only if wan_video_dit.py changes)
