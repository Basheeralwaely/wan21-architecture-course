# Phase 2: DiT Assembly - Pattern Map

**Mapped:** 2026-04-24
**Files analyzed:** 3 (NB-06, NB-07, NB-08 — all new)
**Analogs found:** 3 / 3 (all have exact-role analogs in Phase 1 notebooks + source code)

---

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `Course/NB-06-dit-block.ipynb` | notebook | transform | `Course/NB-05-adaln-zero-modulation.ipynb` | exact (same cell template, same source file subject, most structurally complex Phase 1 notebook) |
| `Course/NB-07-patchify-unpatchify.ipynb` | notebook | transform | `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` | exact (multi-concept notebook, same 3-concept layout) |
| `Course/NB-08-wanmodel-forward.ipynb` | notebook | request-response | `Course/NB-04-self-cross-attention.ipynb` | exact (end-to-end composed forward pass, same prose→code→verify pattern) |

**Source-code analog for all three:** `diffsynth/models/wan_video_dit.py` lines 190–416 (read in full during this session)

---

## Pattern Assignments

### `Course/NB-06-dit-block.ipynb` (notebook, transform)

**Primary analog:** `Course/NB-05-adaln-zero-modulation.ipynb`
**Source code subject:** `diffsynth/models/wan_video_dit.py` lines 190–231 (GateModule, DiTBlock)

---

**Cell 1 — Markdown header pattern** (from NB-05, cell nb05-cell-01):

```markdown
# NB-06: The DiT Block — Assembling Self-Attention, Cross-Attention, and FFN

## Learning Objectives
- Trace DiTBlock.__init__ to see how six sub-modules wire together: self_attn, cross_attn, norm1/2/3, ffn, modulation, gate
- Walk through DiTBlock.forward() line-by-line, connecting each operation to its prior notebook (NB-01 modulate, NB-04 attention, NB-05 adaLN-Zero)
- Count parameters per sub-module and explain why q, k, v, o, ffn.0, ffn.2 are LoRA targets

## Prerequisites
- **Prior notebooks:** NB-01 (modulate), NB-04 (SelfAttention, CrossAttention), NB-05 (adaLN-Zero, GateModule)
- **Assumed concepts:** Residual connections, adaLN-Zero gate=0 identity, 3D RoPE frequency assembly

## Concept Map
- DiTBlock → assembled 30× in WanModel (NB-08)
- DiTBlock.modulation → per-block learned offset on time conditioning (NB-05 established the pattern)
- DiTBlock.forward freqs → requires full 3D assembly (not the simplified single-band shortcut used in NB-04)
```

---

**Cell 2 — Setup cell pattern** (copy verbatim from NB-05 cell nb05-cell-02, adjust imports):

```python
import sys
import types
import importlib.util
import pathlib

# Find project root: walk up from Course/ to find the directory containing diffsynth/
_here = pathlib.Path().resolve()
PROJECT_ROOT = None
for _candidate in [_here, _here.parent, _here.parent.parent]:
    if (_candidate / "diffsynth" / "models" / "wan_video_dit.py").exists():
        PROJECT_ROOT = _candidate
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not find diffsynth/models/wan_video_dit.py. "
        "Run this notebook from Course/ inside the project checkout."
    )
print(f"Project root: {PROJECT_ROOT}")

# Stub wan_video_camera_controller (not needed for DiT primitive demos)
_cam_stub = types.ModuleType("diffsynth.models.wan_video_camera_controller")
_cam_stub.SimpleAdapter = type("SimpleAdapter", (), {"__init__": lambda s, *a, **kw: None})
_diffsynth_stub = types.ModuleType("diffsynth")
_models_stub = types.ModuleType("diffsynth.models")
sys.modules["diffsynth"] = _diffsynth_stub
sys.modules["diffsynth.models"] = _models_stub
sys.modules["diffsynth.models.wan_video_camera_controller"] = _cam_stub

# Load wan_video_dit.py directly, bypassing the broken diffsynth/__init__.py chain
_dit_path = PROJECT_ROOT / "diffsynth" / "models" / "wan_video_dit.py"
_spec = importlib.util.spec_from_file_location("diffsynth.models.wan_video_dit", _dit_path)
dit = importlib.util.module_from_spec(_spec)
sys.modules["diffsynth.models.wan_video_dit"] = dit
_spec.loader.exec_module(dit)

# NB-06 imports
from diffsynth.models.wan_video_dit import (
    DiTBlock, GateModule, modulate, precompute_freqs_cis_3d
)
import torch
import torch.nn as nn

print("Setup complete.")
```

---

**DiTBlock.__init__ sub-module inventory pattern** (source: wan_video_dit.py lines 197–213):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 197-213
dim, num_heads, ffn_dim = 1536, 12, 8960

block = DiTBlock(has_image_input=True, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)

# Sub-module inventory (D-03: parameter counts per sub-module)
print("DiTBlock sub-modules:")
print(f"  self_attn.q  (Linear {dim}→{dim}): {sum(p.numel() for p in block.self_attn.q.parameters()):,} params")
print(f"  self_attn.k  (Linear {dim}→{dim}): {sum(p.numel() for p in block.self_attn.k.parameters()):,} params")
print(f"  self_attn.v  (Linear {dim}→{dim}): {sum(p.numel() for p in block.self_attn.v.parameters()):,} params")
print(f"  self_attn.o  (Linear {dim}→{dim}): {sum(p.numel() for p in block.self_attn.o.parameters()):,} params")
print(f"  cross_attn.q (Linear {dim}→{dim}): {sum(p.numel() for p in block.cross_attn.q.parameters()):,} params")
print(f"  cross_attn.k (Linear {dim}→{dim}): {sum(p.numel() for p in block.cross_attn.k.parameters()):,} params")
print(f"  cross_attn.v (Linear {dim}→{dim}): {sum(p.numel() for p in block.cross_attn.v.parameters()):,} params")
print(f"  cross_attn.o (Linear {dim}→{dim}): {sum(p.numel() for p in block.cross_attn.o.parameters()):,} params")
print(f"  ffn[0] (Linear {dim}→{ffn_dim}): {sum(p.numel() for p in block.ffn[0].parameters()):,} params")
print(f"  ffn[2] (Linear {ffn_dim}→{dim}): {sum(p.numel() for p in block.ffn[2].parameters()):,} params")
print(f"  modulation (nn.Parameter [1,6,{dim}]): {block.modulation.numel():,} params")
total = sum(p.numel() for p in block.parameters())
print(f"  Block total: {total:,}")
```

---

**3D freqs assembly pattern for DiTBlock** (source: wan_video_dit.py lines 381–385, RESEARCH.md Pattern 1):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 381-385 (WanModel.forward)
# IMPORTANT: DiTBlock needs the FULL 3-band freqs — NOT the simplified single-band
# shortcut used in NB-04. See RESEARCH.md Pitfall 1.
head_dim = dim // num_heads  # 128

f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
# f_freqs: complex tensor [1024, 22]  (temporal band: head_dim - 2*(head_dim//3) = 44//2 = 22 complex pairs)
# h_freqs: complex tensor [1024, 21]  (height band:   head_dim//3 = 42//2 = 21 complex pairs)
# w_freqs: complex tensor [1024, 21]  (width band)

F, H, W = 4, 4, 4  # grid dims after patchify with stride (1,2,2): seq_len = F*H*W = 64
freqs = torch.cat([
    f_freqs[:F].view(F, 1, 1, -1).expand(F, H, W, -1),  # [F, H, W, 22] temporal
    h_freqs[:H].view(1, H, 1, -1).expand(F, H, W, -1),  # [F, H, W, 21] height
    w_freqs[:W].view(1, 1, W, -1).expand(F, H, W, -1),  # [F, H, W, 21] width
], dim=-1).reshape(F * H * W, 1, -1)                    # [64, 1, 64]  (22+21+21=64=head_dim//2)
assert freqs.shape == torch.Size([F * H * W, 1, head_dim // 2])
```

---

**DiTBlock forward pass verification pattern** (source: wan_video_dit.py lines 215–231, RESEARCH.md Pattern 1):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 215-231
block.eval()
B, S = 1, F * H * W   # S = 64
x       = torch.randn(B, S, dim)          # [B, S, dim]      — video tokens
context = torch.randn(B, 277, dim)        # [B, 277, dim]    — 257 CLIP + 20 text tokens
t_mod   = torch.randn(B, 6, dim)          # [B, 6, dim]      — time modulation (from WanModel.time_projection)

with torch.no_grad():
    out = block(x, context, t_mod, freqs)  # [B, S, dim]
assert out.shape == torch.Size([B, S, dim])
print(f"DiTBlock output: {out.shape}")    # torch.Size([1, 64, 1536])
```

---

**DiTBlock forward annotated pattern** (source: wan_video_dit.py lines 219–231, RESEARCH.md Code Examples):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 219-231
# Annotated line-by-line for NB-06 prose-before-code cells

# 1. Extract six modulation parameters (recall NB-05: adaLN-Zero)
#    modulation: [1, 6, dim] (learned per-block offset)
#    t_mod:      [B, 6, dim] (from WanModel.time_projection)
#    chunk(6, dim=1) splits [B, 6, dim] into six [B, 1, dim] tensors
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
).chunk(6, dim=1)  # six [B, 1, dim] tensors

# 2. Self-attention branch (recall NB-04: SelfAttention, NB-01: modulate)
input_x = modulate(block.norm1(x), shift_msa, scale_msa)   # adaLN pre-norm
x = block.gate(x, gate_msa, block.self_attn(input_x, freqs))  # gated residual

# 3. Cross-attention branch (recall NB-04: CrossAttention) — no adaLN on cross-attn
x = x + block.cross_attn(block.norm3(x), context)          # simple residual, no gate

# 4. FFN branch (recall NB-01: modulate)
input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)   # adaLN pre-norm
x = block.gate(x, gate_mlp, block.ffn(input_x))            # gated residual
```

---

**LoRA parameter scaling pattern** (D-03, D-04 — per-block then full-model):

```python
# Per-block LoRA target counts (verified: RESEARCH.md Parameter Count Table)
lora_targets = {
    "self_attn.q":  sum(p.numel() for p in block.self_attn.q.parameters()),   # 2,360,832
    "self_attn.k":  sum(p.numel() for p in block.self_attn.k.parameters()),   # 2,360,832
    "self_attn.v":  sum(p.numel() for p in block.self_attn.v.parameters()),   # 2,360,832
    "self_attn.o":  sum(p.numel() for p in block.self_attn.o.parameters()),   # 2,360,832
    "cross_attn.q": sum(p.numel() for p in block.cross_attn.q.parameters()), # 2,360,832
    "cross_attn.k": sum(p.numel() for p in block.cross_attn.k.parameters()), # 2,360,832
    "cross_attn.v": sum(p.numel() for p in block.cross_attn.v.parameters()), # 2,360,832
    "cross_attn.o": sum(p.numel() for p in block.cross_attn.o.parameters()), # 2,360,832
    "ffn.0":        sum(p.numel() for p in block.ffn[0].parameters()),        # 13,771,520
    "ffn.2":        sum(p.numel() for p in block.ffn[2].parameters()),        # 13,764,096
}
block_total = sum(p.numel() for p in block.parameters())          # 51,163,904
lora_total_per_block = sum(lora_targets.values())                 # 46,422,272

print(f"Per-block LoRA parameters: {lora_total_per_block:,} / {block_total:,} = {lora_total_per_block/block_total*100:.1f}%")
num_layers = 30
print(f"Full-model LoRA parameters (×{num_layers} blocks): {lora_total_per_block * num_layers:,}")
# Note: cross_attn.k_img and cross_attn.v_img are NOT LoRA targets
# (default --lora_target_modules "q,k,v,o,ffn.0,ffn.2" — verified from train_lora_cached.py line 292)
```

---

**Summary markdown pattern** (copy structure from NB-05 cell nb05-cell-13):

```markdown
## Summary

### Key Takeaways
- **DiTBlock composition**: six sub-modules (self_attn, cross_attn, norm1/2/3, ffn, modulation, gate) assembled in __init__; forward applies adaLN conditioning twice (before self-attn and before FFN) with a simple residual for cross-attn (line 228: `x = x + block.cross_attn(block.norm3(x), context)`)
- **freqs construction**: DiTBlock.forward requires full 3-band freqs ([seq, 1, head_dim//2]); the single-band shortcut from NB-04 raises RuntimeError inside rope_apply
- **LoRA targets**: q, k, v, o in both self_attn and cross_attn, plus ffn.0 and ffn.2 — 90.7% of block parameters per block, ~1.39B params across the full 30-block model

### Source References
| Symbol | Location |
|--------|---------|
| GateModule | diffsynth/models/wan_video_dit.py, line 190 |
| DiTBlock.__init__ | diffsynth/models/wan_video_dit.py, line 198 |
| DiTBlock.forward | diffsynth/models/wan_video_dit.py, line 215 |
```

---

**Exercise pattern** (copy format from NB-05 cell nb05-cell-14, 3 exercises):

```markdown
## Exercises

### Exercise 1 — Remove cross-attention
Set `has_image_input=False` when creating DiTBlock. Rerun the forward pass with a smaller context tensor (`context = torch.randn(B, 20, dim)`). What changes in the parameter count? (Hint: k_img and v_img disappear from cross_attn.)

### Exercise 2 — Inspect gate values
After the forward pass, manually compute `gate_msa` from the modulation + t_mod combination. Print its mean and std. Is the gate near zero? What would happen if you set gate_msa to exactly zero for a forward pass?

### Exercise 3 — Scale LoRA to different ranks
The LoRA target for self_attn.q has 2,360,832 parameters (dim×dim = 1536×1536). If LoRA uses rank r=16, the adapter has 2×(dim×r) = 2×(1536×16) = 49,152 parameters. Calculate the compression ratio. Repeat for ffn.0 (1536→8960).
```

---

### `Course/NB-07-patchify-unpatchify.ipynb` (notebook, transform)

**Primary analog:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` (multi-concept notebook with 3 distinct concepts in one notebook)
**Source code subject:** `diffsynth/models/wan_video_dit.py` lines 254–270 (Head), 340–357 (patchify/unpatchify)

---

**Cell 1 — Markdown header pattern** (structure from NB-01 cell 712ac433):

```markdown
# NB-07: Patchify, Unpatchify, and the Head Module

## Learning Objectives
- Trace WanModel.patchify: how Conv3d(48→1536, kernel=(1,2,2), stride=(1,2,2)) converts video (B,48,F,H,W) into token sequences (B, F*(H//2)*(W//2), 1536)
- Understand the spatial-to-sequence mapping: one latent frame maps to H//2 × W//2 tokens, temporal dimension preserved (patch_size[0]=1)
- Trace WanModel.unpatchify: how einops rearrange reverses the token sequence back to video shape (B,16,F,H,W)
- See how Head applies adaLN modulation to project from model dim to output latent channels

## Prerequisites
- **Prior notebooks:** NB-01 (modulate function — Head uses it for output conditioning), NB-06 (DiTBlock — patchify produces the token sequence that flows through DiT blocks)
- **Assumed concepts:** Conv3d kernel/stride mechanics, einops rearrange axis labeling

## Concept Map
- WanModel.patchify → converts the 48-channel input into the sequence that DiT blocks process (NB-06, NB-08)
- WanModel.unpatchify → reconstructs output video from the Head's projected tokens (NB-08)
- Head → final output projection applied after all DiT blocks, before unpatchify (NB-08)
```

---

**Cell 2 — Setup cell pattern** (same importlib pattern, adjusted imports):

```python
# Same importlib setup as all Phase 1 notebooks — copy verbatim from NB-05 setup cell
# (omitted here; see Shared Patterns: Import Setup Cell)

# NB-07 specific imports
from diffsynth.models.wan_video_dit import WanModel, Head
import torch
import torch.nn as nn
import math
from einops import rearrange

print("Setup complete.")
```

---

**Patchify pattern** (source: wan_video_dit.py lines 340–350, RESEARCH.md Pattern 2):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 340-350
# patchify = Conv3d projection + rearrange to sequence

patch_size = (1, 2, 2)
in_dim, dim = 48, 1536

patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
# weight shape: (1536, 48, 1, 2, 2) — 1536 filters, each 48-channel × 1×2×2 kernel
# bias shape:   (1536,)

B, C, F, H, W = 1, 48, 4, 8, 8
x = torch.randn(B, C, F, H, W)          # [1, 48, 4, 8, 8]
px = patch_embedding(x)                   # [1, 1536, 4, 4, 4] — stride halves H and W
b, c, f, h, w = px.shape                 # b=1, c=1536, f=4, h=4, w=4

# Source: wan_video_dit.py line 349
x_seq = rearrange(px, 'b c f h w -> b (f h w) c')  # [1, 64, 1536]
assert x_seq.shape == torch.Size([B, f * h * w, dim])
print(f"patchify: {x.shape} -> Conv3d -> {px.shape} -> rearrange -> {x_seq.shape}")
```

---

**Unpatchify pattern** (source: wan_video_dit.py lines 352–357, RESEARCH.md Pattern 2):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 352-357
out_dim = 16
out_dim_full = out_dim * math.prod(patch_size)  # 16 * 1*2*2 = 64

# Simulate head output — shape that Head produces: [B, seq, out_dim*prod(patch_size)]
head_output = torch.randn(B, f * h * w, out_dim_full)  # [1, 64, 64]

# Source: wan_video_dit.py line 353-357 — copy the exact rearrange string
x_video = rearrange(
    head_output, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
    f=f, h=h, w=w, x=patch_size[0], y=patch_size[1], z=patch_size[2]
)  # [1, 16, 4, 8, 8]
assert x_video.shape == torch.Size([B, out_dim, F, H, W])
print(f"unpatchify: {head_output.shape} -> rearrange -> {x_video.shape}")
print(f"Round-trip: {x.shape} -> patchify({x_seq.shape}) -> unpatchify({x_video.shape})")
```

---

**Head module pattern** (source: wan_video_dit.py lines 254–270, RESEARCH.md NB-07 Code Example):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 254-270
dim, out_dim, patch_size = 1536, 16, (1, 2, 2)
head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=1e-6)

print("Head sub-modules:")
print(f"  norm (LayerNorm, no affine): {sum(p.numel() for p in head.norm.parameters()):,} params")
print(f"  head (Linear {dim}→{out_dim * math.prod(patch_size)}): {sum(p.numel() for p in head.head.parameters()):,} params")
print(f"  modulation (nn.Parameter [1,2,{dim}]): {head.modulation.numel():,} params")

B, S = 1, 64
x_seq = torch.randn(B, S, dim)           # [B, seq, dim] — from final DiT block output
t = torch.randn(B, dim)                   # [B, dim] — time embedding BEFORE time_projection
# NOTE: Head receives 't' (output of time_embedding), NOT 't_mod' (the 6-chunk tensor)
# Head applies its own 2-parameter adaLN (shift + scale only, no gate)

head.eval()
with torch.no_grad():
    out = head(x_seq, t)                   # [B, S, out_dim * prod(patch_size)] = [1, 64, 64]
assert out.shape == torch.Size([B, S, out_dim * math.prod(patch_size)])
print(f"Head output: {out.shape}")         # [1, 64, 64]
```

---

**Summary + exercises pattern** (follow NB-01 / NB-05 format):

```markdown
## Summary

### Key Takeaways
- **patchify**: Conv3d(48→1536, kernel=(1,2,2), stride=(1,2,2)) followed by rearrange 'b c f h w -> b (f h w) c'. Temporal patch size=1 means every latent frame becomes its own set of tokens — temporal resolution is fully preserved in token space.
- **unpatchify**: The exact inverse einops string 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)' — always copy from WanModel.unpatchify (line 353) to avoid axis-order bugs.
- **Head**: Applies 2-parameter adaLN (shift + scale, no gate) using the raw time embedding 't', not the 6-chunk 't_mod' used by DiTBlocks. Projects dim→(out_dim * prod(patch_size)) = 1536→64, which unpatchify converts to 16 output channels.

### Source References
| Symbol | Location |
|--------|---------|
| WanModel.patchify | diffsynth/models/wan_video_dit.py, line 340 |
| WanModel.unpatchify | diffsynth/models/wan_video_dit.py, line 352 |
| Head.__init__ | diffsynth/models/wan_video_dit.py, line 255 |
| Head.forward | diffsynth/models/wan_video_dit.py, line 263 |
```

---

### `Course/NB-08-wanmodel-forward.ipynb` (notebook, request-response)

**Primary analog:** `Course/NB-04-self-cross-attention.ipynb` (composed forward pass demo; NB-04 composes self-attn + cross-attn similarly to how NB-08 composes the full WanModel)
**Source code subject:** `diffsynth/models/wan_video_dit.py` lines 273–416 (WanModel)

---

**Cell 1 — Markdown header pattern**:

```markdown
# NB-08: WanModel End-to-End Forward Pass

## Learning Objectives
- See how noise, control, and reference latents are concatenated into the 48-channel input (D-08)
- Trace WanModel.forward from 48-channel video input through patchify, 30 DiT blocks, Head, and unpatchify back to 16-channel output
- Understand gradient checkpointing: training wraps each block in torch.utils.checkpoint.checkpoint() to save VRAM; inference runs blocks directly

## Prerequisites
- **Prior notebooks:** NB-06 (DiTBlock), NB-07 (patchify/unpatchify, Head)
- **Assumed concepts:** Diffusion model latent space, memory-compute tradeoffs in training

## Concept Map
- WanModel → full assembled model; each component is a prior notebook's subject
- 48-channel input → noise (16ch) + control (16ch) + reference (16ch) concatenated on channel dim
- gradient checkpointing → trades ~25% compute for lower peak VRAM during training
```

---

**Cell 2 — Setup cell pattern** (same importlib, adjusted imports):

```python
# Same importlib setup — copy verbatim from NB-05 setup cell
# (see Shared Patterns: Import Setup Cell)

# NB-08 specific imports
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
import torch

print("Setup complete.")
```

---

**48-channel concat demo pattern** (D-08, RESEARCH.md Pattern 3):

```python
# Source: model_architecture.md + diffsynth/models/wan_video_dit.py WanModel.forward
# D-08: show noise/control/ref as SEPARATE tensors before concatenation
B, F, H, W = 1, 4, 8, 8
noise_latent   = torch.randn(B, 16, F, H, W)   # [B, 16, F, H, W] — denoising target
control_latent = torch.randn(B, 16, F, H, W)   # [B, 16, F, H, W] — control video (structure)
ref_latent     = torch.randn(B, 16, F, H, W)   # [B, 16, F, H, W] — reference image (appearance)

x_48 = torch.cat([noise_latent, control_latent, ref_latent], dim=1)  # [B, 48, F, H, W]
assert x_48.shape == torch.Size([B, 48, F, H, W])
print(f"noise:    {noise_latent.shape}")
print(f"control:  {control_latent.shape}")
print(f"ref:      {ref_latent.shape}")
print(f"48-ch x:  {x_48.shape}  <- WanModel.in_dim=48")
```

---

**WanModel reduced-block instantiation pattern** (D-07, RESEARCH.md Pattern 4):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 273-328
# D-07: keep dim=1536 (real architecture), reduce layers to 3 for CPU demo

# --- DEMO CONFIG (3 layers, STD-03 compliant — verified at 0.042s on CPU) ---
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
    num_layers=3,       # REDUCED from 30 for CPU demo
    has_image_input=False,  # avoids needing clip_feature dummy tensor (RESEARCH.md Pitfall 3)
)
# --- PRODUCTION CONFIG (annotation only — do not run on CPU) ---
# model = WanModel(dim=1536, in_dim=48, ffn_dim=8960, out_dim=16,
#     text_dim=4096, freq_dim=256, eps=1e-6, patch_size=(1,2,2),
#     num_heads=12, num_layers=30, has_image_input=True)
```

---

**WanModel forward pass with shape trace pattern** (RESEARCH.md Code Examples NB-08):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 359-416
model.eval()
timestep = torch.tensor([500.0])       # [B] — diffusion timestep
context  = torch.randn(B, 20, 4096)   # [B, L, text_dim=4096] — T5 text embeddings

# Shape trace through WanModel.forward (wan_video_dit.py lines 369-416):
# 1. time_embedding:   timestep [B] -> sinusoidal_embedding_1d -> [B, 256] -> linear+silu+linear -> t [B, 1536]
# 2. time_projection:  t [B, 1536] -> SiLU+linear -> [B, 9216] -> unflatten(1, (6, 1536)) -> t_mod [B, 6, 1536]
# 3. text_embedding:   context [B, 20, 4096] -> 2×linear+GELU -> [B, 20, 1536]
# 4. patchify:         x_48 [B, 48, 4, 8, 8] -> Conv3d(stride=1,2,2) -> [B, 1536, 4, 4, 4] -> rearrange -> [B, 64, 1536]
# 5. freqs assembly:   f_freqs[:4]+h_freqs[:4]+w_freqs[:4] cat on last dim -> [64, 1, 64]
# 6. ×3 DiTBlocks:     x [B, 64, 1536] (shape unchanged through each block)
# 7. head:             x [B, 64, 1536] -> Head.forward(x, t) -> [B, 64, 64]  (64 = 16 * 1*2*2)
# 8. unpatchify:       [B, 64, 64] -> rearrange -> [B, 16, 4, 8, 8]

with torch.no_grad():
    out = model(x_48, timestep, context)   # [B, 16, 4, 8, 8]
assert out.shape == torch.Size([B, 16, F, H, W])
print(f"Input:  {x_48.shape}")
print(f"Output: {out.shape}")
```

---

**Gradient checkpointing pattern** (D-09, RESEARCH.md Pattern 5):

```python
# Source: diffsynth/models/wan_video_dit.py, lines 393-412
# D-09: training vs inference side-by-side

# TRAINING mode — gradient checkpointing active
# IMPORTANT: model.train() MUST be called — checkpointing only activates when self.training=True
# Source: wan_video_dit.py line 393: `if self.training and use_gradient_checkpointing:`
model.train()
x_train = x_48.clone()
out_gc = model(x_train, timestep, context, use_gradient_checkpointing=True)
# Internally for each block: torch.utils.checkpoint.checkpoint(block, x, context, t_mod, freqs, use_reentrant=True)
# -> recomputes block activations during backward instead of storing them -> ~25% lower peak VRAM

# INFERENCE mode — no checkpointing
model.eval()
with torch.no_grad():
    out_eval = model(x_48, timestep, context)
    # Internally: x = block(x, context, t_mod, freqs)  -- direct call, no recomputation

print("Training with checkpointing: saves VRAM by recomputing block activations on backward")
print("Inference without checkpointing: faster, no gradient tracking needed")
assert out_gc.shape == out_eval.shape
print(f"Both paths produce shape: {out_gc.shape}")
```

---

**Summary + exercises pattern** (follow NB-05 summary format):

```markdown
## Summary

### Key Takeaways
- **48-channel input**: noise latent (16ch) + control latent (16ch) + reference latent (16ch) concatenated on dim=1 before entering WanModel. Each 16-channel component comes from a VAE encoder.
- **WanModel.forward data flow**: sinusoidal timestep embedding -> time projection (t_mod) + text embedding -> patchify -> freqs assembly -> N×DiTBlock -> Head -> unpatchify. Every sub-step is a prior notebook's subject.
- **Gradient checkpointing**: requires model.train() — the condition `if self.training and use_gradient_checkpointing` means eval mode always runs direct block calls regardless of the flag.

### Source References
| Symbol | Location |
|--------|---------|
| WanModel.__init__ | diffsynth/models/wan_video_dit.py, line 274 |
| WanModel.forward | diffsynth/models/wan_video_dit.py, line 359 |
| freqs assembly | diffsynth/models/wan_video_dit.py, lines 381-385 |
| checkpoint branch | diffsynth/models/wan_video_dit.py, lines 393-412 |
```

---

## Shared Patterns

### Import Setup Cell (STD-07)
**Source:** `Course/NB-05-adaln-zero-modulation.ipynb` cell nb05-cell-02 (authoritative — most recent Phase 1 notebook; confirmed working via git log)
**Apply to:** ALL three Phase 2 notebooks as Cell 2

The exact working pattern (project-root-walk variant — more robust than the fixed `pathlib.Path("..")` in the PATTERNS.md Phase 1 draft):

```python
import sys
import types
import importlib.util
import pathlib

_here = pathlib.Path().resolve()
PROJECT_ROOT = None
for _candidate in [_here, _here.parent, _here.parent.parent]:
    if (_candidate / "diffsynth" / "models" / "wan_video_dit.py").exists():
        PROJECT_ROOT = _candidate
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not find diffsynth/models/wan_video_dit.py. "
        "Run this notebook from Course/ inside the project checkout."
    )
print(f"Project root: {PROJECT_ROOT}")

_cam_stub = types.ModuleType("diffsynth.models.wan_video_camera_controller")
_cam_stub.SimpleAdapter = type("SimpleAdapter", (), {"__init__": lambda s, *a, **kw: None})
_diffsynth_stub = types.ModuleType("diffsynth")
_models_stub = types.ModuleType("diffsynth.models")
sys.modules["diffsynth"] = _diffsynth_stub
sys.modules["diffsynth.models"] = _models_stub
sys.modules["diffsynth.models.wan_video_camera_controller"] = _cam_stub

_dit_path = PROJECT_ROOT / "diffsynth" / "models" / "wan_video_dit.py"
_spec = importlib.util.spec_from_file_location("diffsynth.models.wan_video_dit", _dit_path)
dit = importlib.util.module_from_spec(_spec)
sys.modules["diffsynth.models.wan_video_dit"] = dit
_spec.loader.exec_module(dit)

# Notebook-specific imports go here
import torch
import torch.nn as nn
print("Setup complete.")
```

---

### Shape Assertion Pattern (STD-04)
**Source:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` cell 265bbe82
**Apply to:** Every code cell that produces a tensor output in all three Phase 2 notebooks

```python
assert output.shape == torch.Size([B, S, dim]), \
    f"Expected ({B}, {S}, {dim}), got {output.shape}"
print(f"Shape OK: {output.shape}")
```

---

### Inline Shape Annotation Pattern (STD-02)
**Source:** `Course/NB-05-adaln-zero-modulation.ipynb` cell nb05-cell-05 (e.g., `gate_zero = torch.zeros(B, 1, dim)  # [B, 1, dim] — broadcasts over S`)
**Apply to:** Every `rearrange` call, every `torch.cat`, every tensor operation with non-obvious output shape

```python
x_seq = rearrange(px, 'b c f h w -> b (f h w) c')  # [B, F*H*W, dim]
x_48  = torch.cat([noise, ctrl, ref], dim=1)         # [B, 48, F, H, W]
freqs = ...reshape(F * H * W, 1, -1)                  # [seq_len, 1, head_dim//2]
```

---

### Source Citation Pattern (STD-06)
**Source:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` and `Course/NB-05-adaln-zero-modulation.ipynb` (consistent across all Phase 1 notebooks)
**Apply to:** First code cell introducing each symbol; also in markdown prose before the cell

Standard citation comment format (copy verbatim):
```python
# Source: diffsynth/models/wan_video_dit.py, lines <N>-<M>
```

Phase 2 source line reference table (STD-06, from RESEARCH.md):

| Symbol | File | Line |
|--------|------|------|
| `GateModule.forward` | `diffsynth/models/wan_video_dit.py` | 194 |
| `DiTBlock.__init__` | `diffsynth/models/wan_video_dit.py` | 198 |
| `DiTBlock.forward` | `diffsynth/models/wan_video_dit.py` | 215 |
| `Head.__init__` | `diffsynth/models/wan_video_dit.py` | 255 |
| `Head.forward` | `diffsynth/models/wan_video_dit.py` | 263 |
| `WanModel.__init__` | `diffsynth/models/wan_video_dit.py` | 274 |
| `WanModel.patchify` | `diffsynth/models/wan_video_dit.py` | 340 |
| `WanModel.unpatchify` | `diffsynth/models/wan_video_dit.py` | 352 |
| `WanModel.forward` | `diffsynth/models/wan_video_dit.py` | 359 |
| freqs assembly | `diffsynth/models/wan_video_dit.py` | 381–385 |
| checkpoint branch | `diffsynth/models/wan_video_dit.py` | 393–412 |

---

### Dummy Tensor Dimensions (STD-03)
**Source:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` cell 265bbe82 + RESEARCH.md Verified Timing Budget
**Apply to:** All code cells creating input tensors in all three Phase 2 notebooks

Standard dims verified within STD-03's 5-second CPU limit:

```python
# NB-06 standard dims
B, S, dim, num_heads = 1, 64, 1536, 12  # S=F*H*W=4*4*4=64; verified 0.020s on CPU

# NB-07 standard dims
B, C, F, H, W = 1, 48, 4, 8, 8          # patchify input; Conv3d produces f=4,h=4,w=4

# NB-08 standard dims
B, F, H, W = 1, 4, 8, 8                 # WanModel demo; verified 0.042s with num_layers=3 on CPU
```

Never use `x.cuda()` or `.to("cuda")`. No real weight loading. All tensors are `torch.randn(...)`.

---

### Back-Reference Prose Pattern
**Source:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` Concept Map section; `Course/NB-05-adaln-zero-modulation.ipynb` Section 5 heading "Connecting modulation Back to the modulate Function (NB-01)"
**Apply to:** All prose markdown cells in all three Phase 2 notebooks when invoking Phase 1 primitives

Format (copy exactly):
```markdown
(recall NB-05: adaLN-Zero — shift_msa, scale_msa, gate_msa come from the six-chunk operation)
(recall NB-04: SelfAttention — input_x flows through self_attn before being gated)
(recall NB-01: modulate — x * (1 + scale) + shift)
```

---

### Exercise Format Pattern (D-03, D-04 per 01-CONTEXT.md)
**Source:** `Course/NB-05-adaln-zero-modulation.ipynb` cell nb05-cell-14 (3 exercises per notebook)
**Apply to:** Final markdown cell in all three Phase 2 notebooks

```markdown
## Exercises

### Exercise 1 — [Concept variation]
[One paragraph describing what to change and what to observe]

### Exercise 2 — [Different concept variation]
[One paragraph]

### Exercise 3 — [Third concept variation]
[One paragraph]
```

---

### ASCII Diagram Pattern (D-02 for NB-06, D-05 for NB-07)
**Source:** No existing notebook uses ASCII diagrams (Phase 1 notebooks use prose + tables). This is a new pattern for Phase 2.
**Apply to:** Top of NB-06 (after header cell, before setup cell or as part of Cell 1); top of NB-07

NB-06 ASCII block diagram (D-02) — place in Cell 1 markdown:
```
DiT Block Data Flow
═══════════════════
         x [B, S, dim]
         │
         ├──► norm1 ──► modulate(shift_msa, scale_msa) ──► SelfAttention ──► × gate_msa ──► + ──► x'
         │                                                   (RoPE via freqs)                  │
         │                                                                                      │
         │◄─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──► norm3 ──► CrossAttention(context) ──────────────────────────────────────────► + ──► x''
         │                                                                                  │
         │◄──────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──► norm2 ──► modulate(shift_mlp, scale_mlp) ──► FFN ──────────────► × gate_mlp ──► + ──► output
         │                                                                                     │
         └◄────────────────────────────────────────────────────────────────────────────────────┘
```

NB-07 ASCII spatial-to-sequence diagram (D-05) — place in patchify section:
```
Spatial-to-Sequence Mapping (patch_size=(1,2,2))
══════════════════════════════════════════════════
Input:  [B, 48, F, H, W]   e.g. [1, 48, 4, 8, 8]

          Frame f=0        Frame f=1         ...  Frame f=3
         ┌────────┐       ┌────────┐              ┌────────┐
         │H=8, W=8│  →    │H=8, W=8│              │H=8, W=8│
         └────────┘       └────────┘              └────────┘
             ↓ Conv3d(stride=1,2,2): halves H and W per frame
         ┌────────┐       ┌────────┐              ┌────────┐
         │ h=4,w=4│       │ h=4,w=4│              │ h=4,w=4│   → 16 tokens per frame
         └────────┘       └────────┘              └────────┘
             ↓ rearrange: 'b c f h w -> b (f h w) c'
         Token sequence: [B, F*h*w, dim] = [1, 64, 1536]
         Position:        [0..15] = frame 0 tokens, [16..31] = frame 1, ...
```

---

## No Analog Found

All Phase 2 notebooks have strong analogs in Phase 1. No file is without a match.

The one new pattern with no Phase 1 precedent is the **ASCII diagram** (D-02, D-05). RESEARCH.md specifies the content; the planner should author the exact ASCII art during notebook construction.

---

## Anti-Patterns (Do Not Use)

All carry forward from Phase 1 `01-PATTERNS.md` Anti-Patterns table, plus Phase 2-specific additions from RESEARCH.md:

| Anti-Pattern | Why Banned | Correct Alternative |
|---|---|---|
| `freqs = f_freqs[:S].unsqueeze(1)` (single-band) in NB-06 | Raises RuntimeError inside rope_apply (last dim 22 ≠ head_dim//2 = 64) | Use full three-band assembly from WanModel.forward lines 381-385 |
| `WanModel(num_layers=30)` for CPU demo | Model init alone takes several seconds; STD-03 violation | Use `num_layers=3` for demo cells; annotate real config in a comment |
| `WanModel(has_image_input=True)` without `clip_feature` | AttributeError in forward — CLIP MLP is called unconditionally | Use `has_image_input=False` for NB-08 demo (RESEARCH.md Pitfall 3) |
| `model.eval()` then `use_gradient_checkpointing=True` | Checkpoint path silently never activates (line 393: `if self.training and ...`) | Call `model.train()` before checkpointing demo, `model.eval()` before inference demo |
| Copying unpatchify einops string from memory | Axis order errors produce wrong spatial layout | Copy exact string from `WanModel.unpatchify` (line 353-356) verbatim |
| `t_mod` vs `t` confusion in Head.forward | Head receives `t` (dim-vector before 6-chunk), not the `t_mod` used by DiTBlocks | Add explicit comment: "Head receives 't', NOT 't_mod' — it applies its own 2-parameter adaLN" |
| Code cell before markdown prose | Violates STD-05 | Always markdown → code → assert ordering |
| `import diffsynth` directly | Triggers modelscope import failure | Use importlib setup cell (Shared Patterns) |

---

## Metadata

**Analog search scope:** `Course/` (5 Phase 1 notebooks read), `diffsynth/models/wan_video_dit.py` (lines 190–416 read), `.planning/phases/01-dit-foundations/01-PATTERNS.md` (read in full)
**Files scanned:** 7 (NB-01, NB-05 notebooks read in full; wan_video_dit.py lines 190–416; 01-PATTERNS.md; 02-CONTEXT.md; 02-RESEARCH.md)
**Source line numbers:** All verified against actual wan_video_dit.py in this session
**Pattern extraction date:** 2026-04-24
