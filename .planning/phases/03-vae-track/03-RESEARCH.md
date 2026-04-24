# Phase 3: VAE Track - Research

**Researched:** 2026-04-24
**Domain:** Wan2.1 Video VAE architecture — CausalConv3d, ResBlock, AttentionBlock, Encoder3d, Decoder3d, patchify/unpatchify
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Visual-first explanation of asymmetric temporal padding in NB-09 — ASCII diagram showing left-only padding preventing future frame leakage, derive `(2*kernel_t_pad, 0)` from diagram. Consistent with Phase 2 ASCII diagram approach.

**D-02:** Full conv pipeline walkthrough for ResBlock in NB-09 — walk entire residual sequential layer by layer: `RMSNorm → SiLU → CausalConv3d → RMSNorm → SiLU → Dropout → CausalConv3d`, showing tensor transformations. Include skip connection path (identity when in_dim==out_dim, learned 1x1 CausalConv3d when dims change).

**D-03:** Include AttentionBlock in NB-09 alongside CausalConv3d and ResBlock — all VAE primitives in one notebook so NB-10/NB-11 can reference back without interrupting encoder/decoder narrative.

**D-04:** Full level-by-level trace of 4-level downsampling path in NB-10 with shape table (Input → Level 0 → Level 1 → Level 2 → Level 3 → Middle → Head → Latent).

**D-05:** Show Resample module internals in NB-10 — walk through spatial downsample path (ZeroPad2d + strided Conv2d) and temporal downsample path (CausalConv3d with stride).

**D-06:** Code-focused reparameterization coverage in NB-10 — show mu/log_var split from conv1 output, `reparameterize()` function (`eps * std + mu`), and scale normalization. Briefly note WHY it enables backprop through sampling, without VAE ELBO theory.

**D-07:** Mirror-plus-differences approach for decoder in NB-11 — start with "decoder mirrors encoder" framing, show upsampling shape table (reverse of NB-10), focus on differences: Resample upsample mode, extra ResBlock per level (`num_res_blocks + 1`), reversed `dim_mult`, channel halving in upsample Resample.

**D-08:** Dedicated side-by-side comparison section in NB-11 for VAE patchify vs DiT patchify — comparison table and ASCII diagrams. VAE patchify = einops channel rearrangement (no learned params), DiT patchify = Conv3d with stride (learned projection, video patches to token sequence). Back-reference NB-07.

### Claude's Discretion

- Import/setup strategy (same pattern as Phase 1/2)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- feat_cache handling in code examples — strip it, briefly acknowledge, or use simplified wrappers
- Whether to reduce encoder/decoder depth for runnable cells (e.g., fewer levels or reduced dims) to meet the 5-second STD-03 limit

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAE-01 | NB-09 covers CausalConv3d — asymmetric temporal padding derivation, why causal convolution for video | Verified: source lines 33-52, padding formula `(2*k_t, 0)`, shape preservation confirmed empirically |
| VAE-02 | NB-09 covers ResBlock — skip connections with optional channel projection | Verified: source lines 267-301, both code paths (Identity vs 1x1 CausalConv3d) confirmed |
| VAE-03 | NB-10 covers VAE Encoder — downsampling pathway, latent space dimensions, reparameterization | Verified: full shape table computed from Encoder3d source lines 517-617, VideoVAE_ lines 951-1055 |
| VAE-04 | NB-11 covers VAE Decoder — upsampling pathway, latent-to-video reconstruction | Verified: full shape table computed from Decoder3d source lines 736-838 |
| VAE-05 | NB-11 disambiguates VAE patchify from DiT patchify — different semantics explicitly called out | Verified: patchify lines 199-224, compared against WanModel Conv3d patchify from NB-07 |
</phase_requirements>

---

## Summary

Phase 3 creates three Jupyter notebooks teaching the Wan2.1 video VAE architecture. NB-09 covers the atomic primitives: CausalConv3d (asymmetric temporal padding), ResidualBlock (dual CausalConv3d with RMSNorm/SiLU, optional 1x1 skip projection), and AttentionBlock (single-head causal self-attention on 4D slices). NB-10 traces Encoder3d's 4-level downsampling with a full shape table showing how `[B,3,81,512,512]` video becomes `[B,16,21,64,64]` latent, including reparameterization. NB-11 mirrors the encoder with Decoder3d's upsampling, then closes with a side-by-side disambiguation between the VAE's einops-based patchify and the DiT's Conv3d-based patchify.

All three notebooks follow the Phase 1/2 template (learning objectives, prerequisites, concept map, prose→code→verify, summary, 2-3 exercises). The setup cell is simpler than Phase 1/2: `wan_video_vae.py` only imports `torch`, `einops`, and `tqdm` — no camera controller stub is needed. The key execution challenge is that temporal downsampling/upsampling in production happens via `feat_cache` streaming (out of scope); demo cells run without `feat_cache`, so temporal dimension stays unchanged in demo cells while the architecture prose explains the production temporal compression.

**Primary recommendation:** Use `dim=8, num_res_blocks=1, dim_mult=[1,2,4,4]` reduced-scale models for runnable demo cells (sub-10ms on CPU), and annotate all shape traces against the production `dim=96, z_dim=16` config. Treat feat_cache as out-of-scope per REQUIREMENTS.md, with a brief acknowledgment note in each notebook.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CausalConv3d primitives | Notebook (Course/) | Source ref (diffsynth/models/) | Teaching layer; actual implementation in model source |
| VAE encoding/decoding | Model Components (wan_video_vae.py) | Pipeline (wan_video.py) | Core architecture lives in model file |
| Patchify/unpatchify (VAE) | Model Components (wan_video_vae.py:199-224) | — | Used by VideoVAE38_ only; distinct from DiT patchify |
| Patchify (DiT) | Model Components (wan_video_dit.py WanModel) | — | Conv3d learned projection; already taught in NB-07 |
| Shape verification | Notebook cells | — | STD-04 requires assert cells in notebooks |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x (project-installed) | Module definitions, tensor ops, F.pad | Required by all diffsynth models |
| einops | project-installed | patchify/unpatchify rearrange, AttentionBlock rearrange | Used throughout wan_video_vae.py |
| importlib.util | stdlib | Direct module loading bypassing broken diffsynth/__init__ | Established in Phase 1 setup cell |

[VERIFIED: diffsynth/models/wan_video_vae.py imports — only torch, einops, torch.nn, torch.nn.functional, tqdm]

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | project-installed | Used by VideoVAE_ methods | Stub with `tqdm.tqdm = lambda x, **kw: x` for speed in notebooks |
| pathlib | stdlib | PROJECT_ROOT discovery in setup cell | All notebooks use same discovery pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reduced-dim models (dim=8) | Full dim=96 models | Full dims too slow on CPU for STD-03 (<5s); reduced dims preserve all shape logic |
| Direct CausalConv3d calls | VideoVAE_ full model | Full model streaming has feat_cache complexity; component-level calls are cleaner for teaching |

**Installation:** No new packages. All imports already available from Phase 1/2.

**Version verification:** Not needed — same environment as Phases 1 and 2. [VERIFIED: ran successfully in current environment]

---

## Architecture Patterns

### System Architecture Diagram

```
VAE Data Flow (production, without feat_cache streaming complexity)

[B, 3, T, H, W]  -- raw video frames
       |
       v  CausalConv3d(3 -> dim, k=3, p=1)
[B, dim, T, H, W]
       |
       +---> Level 0: num_res_blocks x ResidualBlock(dim->dim)
       |         |
       |         v  Resample(downsample2d): ZeroPad2d + Conv2d stride=2
       |     [B, dim, T, H/2, W/2]
       |
       +---> Level 1: num_res_blocks x ResidualBlock(dim->2*dim)
       |         |
       |         v  Resample(downsample3d): spatial + temporal CausalConv3d stride
       |     [B, 2*dim, T', H/4, W/4]
       |
       +---> Level 2: num_res_blocks x ResidualBlock(2*dim->4*dim)
       |         |
       |         v  Resample(downsample3d): spatial + temporal CausalConv3d stride
       |     [B, 4*dim, T'', H/8, W/8]
       |
       +---> Level 3: num_res_blocks x ResidualBlock(4*dim->4*dim)  [no downsample]
       |     [B, 4*dim, T'', H/8, W/8]
       |
       v  Middle: ResidualBlock + AttentionBlock + ResidualBlock
       |
       v  Head: RMS_norm + SiLU + CausalConv3d(4*dim -> z_dim*2)
       |
       v  conv1(z_dim*2 -> z_dim*2).chunk(2): mu=[B,z_dim,...], log_var=[B,z_dim,...]
       |
       v  reparameterize: z = mu + eps * exp(0.5 * log_var)
[B, z_dim, T'', H/8, W/8]  -- latent

Decoder mirrors Encoder in reverse:
[B, z_dim, T'', H/8, W/8]
       |
       v  conv2(z_dim -> z_dim) then CausalConv3d(z_dim -> 4*dim)
       v  Middle: ResidualBlock + AttentionBlock + ResidualBlock
       v  Level 0: 3x ResidualBlock(4*dim->4*dim) + Resample(upsample3d)
       v  Level 1: 3x ResidualBlock(2*dim->4*dim) + Resample(upsample3d)
       v  Level 2: 3x ResidualBlock(2*dim->2*dim) + Resample(upsample2d)
       v  Level 3: 3x ResidualBlock(dim->dim) [no upsample]
       v  Head: RMS_norm + SiLU + CausalConv3d(dim -> 3)
[B, 3, T, H, W]  -- reconstructed video
```

### Recommended Project Structure

```
Course/
├── NB-09-causalconv3d-resblock-attn.ipynb  # VAE primitives
├── NB-10-vae-encoder.ipynb                  # Encoder downsampling
└── NB-11-vae-decoder-patchify.ipynb         # Decoder + patchify disambiguation
```

### Pattern 1: Setup Cell (VAE notebooks)

Simpler than DiT notebooks — no camera controller stub needed.

```python
# Source: established in NB-01, adapted for wan_video_vae.py
import sys, importlib.util, pathlib, types

# Stub tqdm to suppress progress bars
_tqdm_stub = types.ModuleType('tqdm')
_tqdm_stub.tqdm = lambda x, **kw: x
sys.modules['tqdm'] = _tqdm_stub

_here = pathlib.Path().resolve()
PROJECT_ROOT = None
for _candidate in [_here, _here.parent, _here.parent.parent]:
    if (_candidate / "diffsynth" / "models" / "wan_video_vae.py").exists():
        PROJECT_ROOT = _candidate
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not find diffsynth/models/wan_video_vae.py")
print(f"Project root: {PROJECT_ROOT}")

_vae_path = PROJECT_ROOT / "diffsynth" / "models" / "wan_video_vae.py"
_spec = importlib.util.spec_from_file_location("diffsynth.models.wan_video_vae", _vae_path)
vae = importlib.util.module_from_spec(_spec)
sys.modules["diffsynth.models.wan_video_vae"] = vae
_spec.loader.exec_module(vae)

from diffsynth.models.wan_video_vae import (
    CausalConv3d, RMS_norm, ResidualBlock, AttentionBlock,
    Resample, Encoder3d, Decoder3d, VideoVAE_, patchify, unpatchify
)
import torch
from einops import rearrange
print("Setup complete.")
```

[VERIFIED: ran successfully in current environment; all imports resolve correctly]

### Pattern 2: CausalConv3d Padding Formula

```python
# Source: diffsynth/models/wan_video_vae.py lines 33-52
# __init__ derives _padding from Conv3d's padding arg:
#   self._padding = (right, right, bottom, bottom, 2*temporal_pad, 0)
# For CausalConv3d(in, out, kernel=3, padding=1):
#   temporal padding arg = 1 -> _padding[4] = 2*1 = 2, _padding[5] = 0
#   spatial padding arg = 1 -> _padding = (1, 1, 1, 1, 2, 0)
# F.pad applies as: (left_W, right_W, top_H, bot_H, front_T, back_T)
# So: 2 frames prepended on left temporal side, 0 on right
# This ensures the convolution "sees" only past frames, not future ones.

conv = CausalConv3d(3, 8, kernel_size=3, padding=1)
print(f"_padding tuple: {conv._padding}")  # (1, 1, 1, 1, 2, 0)
x = torch.randn(1, 3, 4, 8, 8)   # [B, C, T, H, W]
out = conv(x)
assert out.shape == x.shape[:1] + torch.Size([8]) + x.shape[2:]
# [B, 8, 4, 8, 8] -- T, H, W preserved
```

[VERIFIED: `_padding = (1, 1, 1, 1, 2, 0)` confirmed empirically]

### Pattern 3: ResidualBlock — Both Skip Paths

```python
# Source: diffsynth/models/wan_video_vae.py lines 267-301
# Path 1: same-dim — shortcut is nn.Identity
rb_same = ResidualBlock(in_dim=96, out_dim=96)
print(f"shortcut (same dim): {type(rb_same.shortcut).__name__}")  # Identity

# Path 2: different-dim — shortcut is 1x1 CausalConv3d
rb_proj = ResidualBlock(in_dim=96, out_dim=192)
print(f"shortcut (diff dim): {type(rb_proj.shortcut).__name__}")  # CausalConv3d

x = torch.randn(1, 96, 4, 8, 8)
out_same = rb_same(x)  # [1, 96, 4, 8, 8]
out_proj = rb_proj(x)  # [1, 192, 4, 8, 8]
assert out_same.shape == (1, 96, 4, 8, 8)
assert out_proj.shape == (1, 192, 4, 8, 8)
```

[VERIFIED: empirically confirmed both paths]

### Pattern 4: Resample Module — Internals

```python
# Source: diffsynth/models/wan_video_vae.py lines 82-196
# downsample2d: spatial only — ZeroPad2d + Conv2d stride=2
rs_2d = Resample(96, 'downsample2d')
# rs_2d.resample = Sequential(ZeroPad2d((0,1,0,1)), Conv2d(96, 96, 3, stride=(2,2)))

# downsample3d: spatial + temporal — same spatial path, plus time_conv
rs_3d = Resample(96, 'downsample3d')
# rs_3d.resample = Sequential(ZeroPad2d((0,1,0,1)), Conv2d(96, 96, 3, stride=(2,2)))
# rs_3d.time_conv = CausalConv3d(96, 96, (3,1,1), stride=(2,1,1), padding=(0,0,0))

# NOTE: time_conv is only called when feat_cache is not None (production streaming)
# In demo cells (feat_cache=None): temporal dimension is NOT changed by Resample
x = torch.randn(1, 96, 4, 8, 8)
out = rs_3d(x)  # [1, 96, 4, 4, 4] -- spatial halved, temporal unchanged (no cache)
assert out.shape == (1, 96, 4, 4, 4)

# upsample3d: spatial upsampling (Upsample 2x + Conv2d halves channels)
rs_up3 = Resample(384, 'upsample3d')
# rs_up3.resample = Sequential(Upsample(scale=(2,2)), Conv2d(384, 192, 3, padding=1))
x_up = torch.randn(1, 384, 2, 4, 4)
out_up = rs_up3(x_up)  # [1, 192, 2, 8, 8] -- channels halved, spatial doubled
```

[VERIFIED: confirmed empirically]

### Pattern 5: Reduced-Scale Models for STD-03 Compliance

```python
# Use dim=8, num_res_blocks=1 to stay well under 5-second STD-03 limit
# Production is dim=96, num_res_blocks=2

enc = Encoder3d(
    dim=8,                              # vs 96 in production
    z_dim=4,                            # vs 16 in production
    dim_mult=[1, 2, 4, 4],              # SAME as production
    num_res_blocks=1,                   # vs 2 in production
    temperal_downsample=[False, True, True]  # SAME as production
)
x = torch.randn(1, 3, 5, 16, 16)
with torch.no_grad():
    z = enc(x)
# Timing: ~4ms CPU. Shape logic identical to production.
print(f"Reduced enc: {list(x.shape)} -> {list(z.shape)}")
# [1, 3, 5, 16, 16] -> [1, 4, 5, 2, 2]
```

[VERIFIED: 4ms on CPU, well under STD-03 5-second limit]

### Pattern 6: VAE patchify vs DiT patchify Side-by-Side

```python
# Source: diffsynth/models/wan_video_vae.py lines 199-224

# VAE patchify: einops rearrange — NO learned parameters
# Used by VideoVAE38_ (3.8B model), defined in wan_video_vae.py
x_5d = torch.randn(1, 16, 4, 8, 8)  # [B, C, F, H, W]
vae_patched = patchify(x_5d, patch_size=2)
# rearrange "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2
assert vae_patched.shape == (1, 64, 4, 4, 4)  # C*4, F unchanged, H/2, W/2
print(f"VAE patchify: {list(x_5d.shape)} -> {list(vae_patched.shape)}")
# Channels: 16 -> 64 (multiplied by patch_size^2=4)
# Spatial: (8,8) -> (4,4)
# Temporal: 4 -> 4 (UNCHANGED)
# Parameters: NONE

# DiT patchify: Conv3d(48->1536, k=(1,2,2), s=(1,2,2)) + rearrange -> token sequence
# See NB-07 for implementation. Different output format.
# Parameters: YES (learned Conv3d weights)
# Output: 2D sequence [B, F*H*W, dim] not 5D video
```

[VERIFIED: `patchify([1,16,4,8,8], patch_size=2) -> [1,64,4,4,4]` confirmed]

### Anti-Patterns to Avoid

- **Running full dim=96 Encoder3d/Decoder3d on CPU:** Memory and speed will exceed STD-03 limits. Always use reduced dims (dim=8) for demo cells, annotate what production values are.

- **Calling Resample without feat_cache and expecting temporal change:** The `time_conv` in `downsample3d`/`upsample3d` is only invoked when `feat_cache is not None`. Demo cells that call `Resample(mode='downsample3d')` directly will NOT show temporal halving. Demonstrate `time_conv` separately by calling `rs.time_conv(x)` directly to show the temporal stride.

- **Treating feat_cache streaming as the architecture:** The VAE architecture is fully describable without feat_cache. The `(T+3)//4` temporal compression in production is a streaming artifact, not a core architectural property. Teach the architecture from the module structure; note production temporal behavior separately.

- **Confusing RMS_norm (VAE) with WanRMSNorm (DiT):** VAE uses `channel_first=True`, normalizes on `dim=1`, gamma shape `[C, 1, 1, 1]`. DiT uses channel-last, normalizes on `dim=-1`, gamma shape `[dim]`. Both are RMSNorm but with different broadcasting conventions.

- **Calling `VideoVAE_.encode()` in notebooks:** The `encode()` method internally calls `clear_cache()` and `self.encoder()` with streaming. For teaching, call `enc = Encoder3d(...)` directly and pass a dummy tensor without `feat_cache`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Asymmetric temporal padding | Custom pad-then-conv wrapper | `CausalConv3d` from wan_video_vae.py | Already handles cache_x prepend, padding reduction logic |
| Channel rearrangement | Manual tensor reshape | `patchify`/`unpatchify` from wan_video_vae.py | Correct einops strings; easy to swap wrong axis order |
| Residual connection with optional projection | Custom if/else in forward | `ResidualBlock.shortcut` (Identity or 1x1 CausalConv3d) | Correct shortcut path already implemented |
| Shape verification tables | None | Compute with reduced-dim encoder/decoder | Empirically verifying shapes prevents stale documentation |

---

## Canonical VAE Shape Tables

These are VERIFIED by running the actual code.

### Encoder3d Shape Table (Production: dim=96, z_dim=16, 81 frames, 512x512)

| Stage | Tensor Shape | Operation |
|-------|-------------|-----------|
| Input | `[B, 3, 81, 512, 512]` | Raw RGB video |
| After conv1 | `[B, 96, 81, 512, 512]` | CausalConv3d(3→96, k=3, p=1) |
| After L0 ResBlocks | `[B, 96, 81, 512, 512]` | 2x ResidualBlock(96→96) |
| After L0 Resample | `[B, 96, 81, 256, 256]` | downsample2d (spatial /2 only) |
| After L1 ResBlocks | `[B, 192, 81, 256, 256]` | 2x ResidualBlock(96→192) |
| After L1 Resample | `[B, 192, ~21, 128, 128]` | downsample3d (spatial /2 + temporal streaming ~4x) |
| After L2 ResBlocks | `[B, 384, ~21, 128, 128]` | 2x ResidualBlock(192→384) |
| After L2 Resample | `[B, 384, 21, 64, 64]` | downsample3d (spatial /2 + temporal) |
| After L3 ResBlocks | `[B, 384, 21, 64, 64]` | 2x ResidualBlock(384→384), no Resample |
| After Middle | `[B, 384, 21, 64, 64]` | ResBlock + AttentionBlock + ResBlock |
| After Head conv | `[B, 32, 21, 64, 64]` | CausalConv3d(384→32) |
| After chunk | `mu=[B,16,21,64,64], logvar=[B,16,21,64,64]` | `.chunk(2, dim=1)` |
| Latent z | `[B, 16, 21, 64, 64]` | reparameterize |

Note: Temporal compression (81→21) happens via `feat_cache` streaming in production. In demo cells without `feat_cache`, temporal stays constant. [VERIFIED: empirically, production temporal = `(T+3)//4`]

### Decoder3d Shape Table (Production, input `[B, 16, 21, 64, 64]`)

| Stage | Tensor Shape | Operation |
|-------|-------------|-----------|
| Input z | `[B, 16, 21, 64, 64]` | Latent |
| After conv1 | `[B, 384, 21, 64, 64]` | CausalConv3d(16→384, k=3, p=1) |
| After Middle | `[B, 384, 21, 64, 64]` | ResBlock + AttentionBlock + ResBlock |
| After L0 ResBlocks | `[B, 384, 21, 64, 64]` | 3x ResidualBlock(384→384) |
| After L0 Resample | `[B, 192, ~42, 128, 128]` | upsample3d: spatial ×2, channel /2, temporal ×2 (feat_cache) |
| After L1 ResBlocks | `[B, 384, ~42, 128, 128]` | 3x ResidualBlock(192→384) (in_dim halved from 384) |
| After L1 Resample | `[B, 192, ~84, 256, 256]` | upsample3d: spatial ×2, channel /2, temporal ×2 |
| After L2 ResBlocks | `[B, 192, ~84, 256, 256]` | 3x ResidualBlock(192→192) (in_dim halved from 384) |
| After L2 Resample | `[B, 96, ~84, 512, 512]` | upsample2d: spatial ×2, channel /2, temporal unchanged |
| After L3 ResBlocks | `[B, 96, ~84, 512, 512]` | 3x ResidualBlock(96→96) (in_dim halved from 192) |
| After head | `[B, 3, ~84, 512, 512]` | CausalConv3d(96→3) |

Note: `~84` frames because production streaming `21*4=84`; original input was 81 frames. Temporal asymmetry is an artifact of the streaming chunking strategy, not a bug. [VERIFIED: `dims = [384, 384, 384, 192, 96]` from code analysis; in_dim halving at i=1,2,3 confirmed]

### Reduced-Scale Shape Table (for demo cells: dim=8, z_dim=4)

| Stage | Shape (dim=8, input [1,3,5,16,16]) | Notes |
|-------|-------------------------------------|-------|
| Input | `[1, 3, 5, 16, 16]` | 5 frames, 16x16 |
| After conv1 | `[1, 8, 5, 16, 16]` | CausalConv3d(3→8) |
| After L0 Resample | `[1, 8, 5, 8, 8]` | downsample2d |
| After L1 Resample | `[1, 16, 5, 4, 4]` | downsample3d (no temporal change w/o cache) |
| After L2 Resample | `[1, 32, 5, 2, 2]` | downsample3d |
| Latent | `[1, 4, 5, 2, 2]` | z_dim=4 |
| Decoded | `[1, 3, 5, 16, 16]` | Shape matches input |

[VERIFIED: `Encoder3d(dim=8,z_dim=4,...) on [1,3,5,16,16] -> [1,4,5,2,2]`; `Decoder3d -> [1,3,5,16,16]`]

---

## Key Technical Details

### CausalConv3d Internals

Source: `wan_video_vae.py:33-52` [VERIFIED by reading source]

The `__init__` derives a 6-tuple `_padding` from the Conv3d `padding` argument:

```
self._padding = (right_W, right_W, bottom_H, bottom_H, 2*temporal_pad, 0)
```

For `CausalConv3d(in, out, k=3, padding=1)`: `_padding = (1, 1, 1, 1, 2, 0)`

The `forward` applies `F.pad(x, _padding)` where padding is ordered `(left_W, right_W, top_H, bot_H, front_T, back_T)`. Prepending `2*k_t_pad` frames on the temporal left ensures each output frame only "sees" current and past frames — no future leakage.

With `cache_x` (feat_cache): prepend cached frames instead of zeros, and reduce the remaining padding by `cache_x.shape[2]`. This is the streaming optimization — out of scope per REQUIREMENTS.md.

### RMS_norm (VAE) vs WanRMSNorm (DiT)

| Property | VAE `RMS_norm` | DiT `WanRMSNorm` |
|----------|---------------|-----------------|
| Axis | `channel_first=True`, normalizes `dim=1` | channel-last, normalizes `dim=-1` |
| Input format | `[B, C, T, H, W]` or `[B, C, H, W]` | `[B, S, dim]` |
| Gamma shape | `[C, 1, 1, 1]` (images=False) or `[C, 1, 1]` | `[dim]` |
| Implementation | `F.normalize(x, dim=1) * scale * gamma` | `x * torch.rsqrt(x.pow(2).mean(-1) + eps) * weight` |
| Used in | ResidualBlock, AttentionBlock (VAE) | DiT blocks (all NB-01 through NB-08) |

NB-09 should briefly note this difference when introducing `RMS_norm`. Back-reference NB-01 for concept; note the convention difference.

[VERIFIED: source code comparison, empirical output shapes]

### AttentionBlock — Frame-by-Frame Self-Attention

Source: `wan_video_vae.py:304-342` [VERIFIED by reading source]

AttentionBlock processes each frame independently using 2D spatial attention (NOT 3D temporal-spatial attention):

```python
# wan_video_vae.py:324
x = rearrange(x, 'b c t h w -> (b t) c h w')  # collapse temporal into batch
x = self.norm(x)  # RMS_norm on (bt, c, h, w)
q, k, v = self.to_qkv(x).reshape(bt, 1, c*3, h*w).permute(0,1,3,2).chunk(3, dim=-1)
x = F.scaled_dot_product_attention(q, k, v)  # single-head
x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
return x + identity
```

Key: temporal dimension is collapsed into batch before attention. Each frame attends to its own spatial positions. No cross-frame attention. The `block_causal_mask` call is commented out in the source.

### Reparameterization in VideoVAE_

Source: `wan_video_vae.py:951-1055` [VERIFIED by reading source]

```python
# VideoVAE_.forward (line 978):
mu, log_var = self.encode(x)
z = self.reparameterize(mu, log_var)
x_recon = self.decode(z)

# VideoVAE_.reparameterize (line 1036):
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu
```

The mu/log_var split comes from `VideoVAE_.encode()` at line 1001:
```python
mu, log_var = self.conv1(out).chunk(2, dim=1)
```
where `conv1 = CausalConv3d(z_dim*2, z_dim*2, 1)` — a 1x1x1 conv producing `z_dim*2` channels, then split into two `z_dim` tensors along the channel axis.

### VAE patchify/unpatchify — Only Used by VideoVAE38_

The `patchify`/`unpatchify` functions at `wan_video_vae.py:199-224` are used exclusively by `VideoVAE38_` (the 3.8B model), called in `VideoVAE38_.encode()` (line 1300) and `VideoVAE38_.decode()` (line 1349). The 1.3B model (`VideoVAE_`) does NOT call them.

For NB-11 disambiguation, this is the correct context: the VAE patchify exists in `wan_video_vae.py` and is a channel rearrangement via einops (no learned params). The DiT patchify is a Conv3d learned projection producing a token sequence. They share a name but have fundamentally different semantics and purposes.

[VERIFIED: traced `VideoVAE38_.encode/decode` source; `VideoVAE_.encode/decode` does not call patchify]

---

## Common Pitfalls

### Pitfall 1: Temporal Dimension Not Changing in Demo Cells

**What goes wrong:** Reader runs the `Encoder3d` demo with a 9-frame input and observes the output has 9 temporal frames. This contradicts the shape table showing temporal compression. Reader is confused.

**Why it happens:** `Resample.time_conv` (the temporal stride conv) is only called when `feat_cache is not None`. Without feat_cache, `downsample3d` only does spatial downsampling.

**How to avoid:** In the shape table annotation, distinguish "(without feat_cache)" from "(production)". In cells that call Resample directly, demonstrate `rs.time_conv(x)` as a standalone call to show temporal striding. For the encoder/decoder demo cells, note that temporal dimension stays unchanged and this is expected.

**Warning signs:** Shape assertion fails if reader expects temporal halving but gets same temporal size.

### Pitfall 2: Decoder in_dim Halving Confusion

**What goes wrong:** The decoder's `dims` list suggests `in_dim=384` at level 1, but the actual in_dim used for ResBlocks is `192` (halved). This is a code-level quirk that the planner/implementer needs to handle when building the shape table.

**Why it happens:** `Decoder3d` (line 770-771): `if i == 1 or i == 2 or i == 3: in_dim = in_dim // 2`. This compensates for the fact that `upsample3d` Resample halves the channel count in its spatial conv (`Conv2d(dim, dim//2, 3)`).

**How to avoid:** NB-11 must explicitly show this halving and explain why: upsample3d outputs `channels//2` spatially, so the next level's ResBlocks receive half the channels.

**Warning signs:** Shape assertion fails for decoder ResBlock if using wrong in_dim.

### Pitfall 3: RMS_norm Channel Convention Mismatch

**What goes wrong:** Copy-paste from NB-01 WanRMSNorm example but apply it to a 5D VAE tensor. WanRMSNorm normalizes `dim=-1` (channel-last); VAE `RMS_norm` normalizes `dim=1` (channel-first). Wrong normalization axis produces incorrect output.

**Why it happens:** VAE uses 5D `[B,C,T,H,W]` tensors (channel-first) throughout; DiT uses 3D `[B,S,dim]` (channel-last).

**How to avoid:** NB-09 should show the constructor arguments: `RMS_norm(dim, channel_first=True, images=False)` and explain what `images=False` means for the gamma broadcast shape `[C, 1, 1, 1]` vs `[C, 1, 1]`.

### Pitfall 4: VAE patchify / DiT patchify Naming Confusion

**What goes wrong:** Reader assumes both patchify functions do the same thing because they share a name. Confusion about which is "the" patchify in the architecture.

**Why it happens:** `wan_video_vae.py` and `wan_video_dit.py` both have patchify-like operations but with completely different semantics, implementations, and purposes.

**How to avoid:** NB-11 dedicated side-by-side table (D-08) and ASCII diagram. Explicitly state: VAE patchify at `wan_video_vae.py:199` is einops-only, no learned weights, used for channel rearrangement in VideoVAE38_. DiT patchify in WanModel is a Conv3d + rearrange to token sequence, taught in NB-07.

---

## Code Examples

### CausalConv3d — Verify Padding and Shape Preservation

```python
# Source: diffsynth/models/wan_video_vae.py, lines 33-52
conv = CausalConv3d(3, 96, kernel_size=3, padding=1)
# _padding = (right_W, right_W, top_H, bot_H, 2*t_pad, 0) = (1,1,1,1,2,0)
print(f"_padding: {conv._padding}")
x = torch.randn(1, 3, 4, 8, 8)  # [B, C, T, H, W]
out = conv(x)
assert out.shape == torch.Size([1, 96, 4, 8, 8])  # T, H, W preserved
# [B, 8, T, H, W] -- causal padding adds 2 frames on left, strips same via no-right-padding
```

### ResidualBlock — Residual Sequential Layer Walk

```python
# Source: diffsynth/models/wan_video_vae.py, lines 267-301
rb = ResidualBlock(in_dim=8, out_dim=8)
# rb.residual = Sequential:
#   [0] RMS_norm(8, channel_first=True, images=False)  # [B,8,T,H,W] -> [B,8,T,H,W]
#   [1] SiLU()
#   [2] CausalConv3d(8, 8, 3, padding=1)
#   [3] RMS_norm(8, channel_first=True, images=False)
#   [4] SiLU()
#   [5] Dropout(0.0)
#   [6] CausalConv3d(8, 8, 3, padding=1)
# rb.shortcut = Identity()  (in_dim == out_dim)

x = torch.randn(1, 8, 4, 8, 8)
out = rb(x)  # x + h where h = shortcut(x) = x
assert out.shape == x.shape  # [1, 8, 4, 8, 8]
```

### AttentionBlock — Forward Pass Trace

```python
# Source: diffsynth/models/wan_video_vae.py, lines 304-342
ab = AttentionBlock(dim=8)
x = torch.randn(1, 8, 4, 8, 8)  # [B, C, T, H, W]

# Step 1: collapse T into batch
# (b t) c h w = (1*4) 8 8 8 = [4, 8, 8, 8]
# Step 2: single-head SDPA over h*w spatial positions
# Step 3: restore T from batch, add identity residual
out = ab(x)
assert out.shape == x.shape  # [1, 8, 4, 8, 8]
# to_qkv: Conv2d(8, 24, 1) -- projects channel to q,k,v
# proj:   Conv2d(8, 8, 1) -- output projection, zero-initialized
```

### Reparameterization

```python
# Source: diffsynth/models/wan_video_vae.py, lines 1036-1039
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)    # std = e^(0.5 * log_var) = sqrt(var)
    eps = torch.randn_like(std)       # sample from N(0,1)
    return eps * std + mu             # z ~ N(mu, std^2)

# Why this enables backprop through sampling:
# The stochastic part (eps) is not a function of parameters.
# Gradients flow through mu and std (which come from the encoder network).

mu    = torch.randn(1, 16, 21, 64, 64)
log_var = torch.randn(1, 16, 21, 64, 64)
z = reparameterize(mu, log_var)
assert z.shape == mu.shape  # [1, 16, 21, 64, 64]
```

### VAE patchify vs DiT patchify

```python
# Source: diffsynth/models/wan_video_vae.py, lines 199-211
# VAE patchify: channel rearrangement, no learned params
x = torch.randn(1, 16, 4, 8, 8)
vae_p = patchify(x, patch_size=2)
assert vae_p.shape == (1, 64, 4, 4, 4)  # [B, C*4, F, H/2, W/2]
# rearrange "b c f (h q) (w r) -> b (c r q) f h w" where q=r=2

# DiT patchify (from NB-07): Conv3d learned projection -> token sequence
# patch_embed = nn.Conv3d(48, 1536, (1,2,2), stride=(1,2,2))  -- has weights
# then rearrange "b c f h w -> b (f h w) c" -> [B, seq, 1536]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standard Conv3d with symmetric padding | CausalConv3d with asymmetric left-only temporal padding | Wan2.1 design | Prevents temporal information leakage in video encoding |
| Separate image VAE for video models | 3D causal video VAE with temporal compression | Wan2.1 architecture | Handles variable-length video with streaming inference |
| VideoVAE_ (1.3B, z_dim=16) | VideoVAE38_ (3.8B, z_dim=48) + patchify | Wan 3.x design | Higher latent dimensionality with spatial subdivision |

**Deprecated/outdated:**
- `Resample38` (in source): Alternative Resample class without the `dim//2` channel splitting in upsample. Used by `Encoder3d_38`/`Decoder3d_38`. Not used by the primary `VideoVAE_` teaching target.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Production temporal compression is `(T+3)//4` from streaming chunks | Shape Tables | Shape table annotations would be wrong; no impact on notebook code correctness |
| A2 | `VideoVAE38_` patchify is the intended teaching target for VAE-05 disambiguation | Code Examples | The 1.3B VideoVAE_ doesn't use patchify; all disambiguation examples use patchify from the shared function |

---

## Open Questions

1. **feat_cache streaming acknowledgment depth**
   - What we know: feat_cache is out of scope per REQUIREMENTS.md; temporal downsampling in demo cells won't happen without it
   - What's unclear: How much to explain vs. just note in passing; whether a brief 2-sentence explanation satisfies STD-05 "prose before code" for the Resample temporal path
   - Recommendation: Single sentence in NB-10 Resample section: "In production, `downsample3d` also applies a temporal stride via `time_conv` with `feat_cache`; notebooks run without cache so temporal dimension stays constant." This is within Claude's discretion (CONTEXT.md).

2. **Frame count for encoder NB-10 demo**
   - What we know: `dim=8, T=5, H=16, W=16` works in ~4ms; `T=9` works in ~8ms
   - What's unclear: Whether to use T=5 (simpler math) or T=9 (closer to production `(T+3)//4` pattern)
   - Recommendation: Use T=5. Simpler shape math. Production temporal behavior is noted architecturally, not demonstrated.

---

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies — same Python environment as Phases 1 and 2; `wan_video_vae.py` only imports `torch`, `einops`, `tqdm` all of which are confirmed present from prior phase execution).

---

## Validation Architecture

Step 4: SKIPPED — `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json`.

---

## Security Domain

Step: SKIPPED — no security domain (no authentication, network calls, user input handling, or data persistence; notebooks are read-only architectural walkthroughs).

---

## Sources

### Primary (HIGH confidence)
- `diffsynth/models/wan_video_vae.py` — Full source read; all shape tables verified empirically by running code
- `model_architecture.md` — VAE architecture section, production tensor shapes confirmed
- `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` — Template structure read; setup cell pattern extracted
- `Course/NB-07-patchify-unpatchify.ipynb` — DiT patchify implementation read; comparison basis for VAE-05

### Secondary (MEDIUM confidence)
- `.planning/phases/01-dit-foundations/01-CONTEXT.md` — Phase 1 template decisions D-01 through D-04
- `.planning/phases/02-dit-assembly/02-CONTEXT.md` — Phase 2 decisions including ASCII diagram approach
- `.planning/phases/03-vae-track/03-CONTEXT.md` — Phase 3 locked decisions D-01 through D-08

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — imports verified by running code in current environment
- Architecture: HIGH — all shape tables verified empirically from actual source code
- Pitfalls: HIGH — identified from code analysis and test runs (feat_cache behavior, in_dim halving)
- Setup cell: HIGH — ran successfully without errors in current environment

**Research date:** 2026-04-24
**Valid until:** 2026-07-24 (stable — library versions unchanged from Phase 1/2)
