# Phase 1: DiT Foundations - Pattern Map

**Mapped:** 2026-04-24
**Files analyzed:** 5 (NB-01 through NB-05, all new)
**Analogs found:** 0 / 5 (no prior notebooks exist; source code is the analog)

---

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` | notebook | transform | `diffsynth/models/wan_video_dit.py` lines 64–112 | source-code analog |
| `Course/NB-02-qkv-projections-head-layout.ipynb` | notebook | transform | `diffsynth/models/wan_video_dit.py` lines 125–148 | source-code analog |
| `Course/NB-03-3d-rope.ipynb` | notebook | transform | `diffsynth/models/wan_video_dit.py` lines 75–98 | source-code analog |
| `Course/NB-04-self-cross-attention.ipynb` | notebook | request-response | `diffsynth/models/wan_video_dit.py` lines 125–187 | source-code analog |
| `Course/NB-05-adaln-zero-modulation.ipynb` | notebook | event-driven | `diffsynth/models/wan_video_dit.py` lines 190–231 | source-code analog |

**Note:** No existing `.ipynb` files exist in this repository. The `Course/` directory is empty. The analog for every notebook is the source implementation in `diffsynth/models/wan_video_dit.py`, from which notebooks import and demonstrate real symbols (STD-07).

---

## Pattern Assignments

### All notebooks: Standard Cell Template (D-01, STD-05)

Every notebook MUST follow this cell ordering without exception:

```
Cell 1  [Markdown]  — Title + Learning Objectives (3+ bullets) + Prerequisites + Concept Map
Cell 2  [Code]      — Setup: importlib load of wan_video_dit.py (see Shared Pattern: Import Setup)
Cell N  [Markdown]  — Section heading + prose explanation of Concept A
Cell N+1 [Code]     — Dummy tensors + Concept A operation with inline shape annotations (STD-02)
Cell N+2 [Code]     — assert output.shape == expected (STD-04)
         ...repeat for each concept in the notebook...
Final   [Markdown]  — ## Summary (3-5 key takeaways) + ## Exercises (2-3 modification tasks, D-03)
```

**Concept map format (D-02):** Plain-text list only, e.g.:
```
- RMSNorm → used in SelfAttention q/k normalization (NB-04) and DiT block pre-norms (NB-06)
- sinusoidal_embedding_1d → used in WanModel.time_embedding (NB-06)
- modulate → used in adaLN-Zero conditioning (NB-05) and DiTBlock forward (NB-06)
```

**Exercise format (D-03, D-04):** 2-3 per notebook, each modifying a different concept:
```markdown
### Exercise 1 — Swap normalization
Replace `RMSNorm` with `nn.LayerNorm(dim)` in the forward pass.
Run the shape verification cell. What changes? What stays the same?
Compare the parameter counts (`norm.weight` vs `ln.weight`, `ln.bias`).
```

---

### `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` (notebook, transform)

**Analog:** `diffsynth/models/wan_video_dit.py`
**Covers:** DIT-01 (RMSNorm), DIT-02 (sinusoidal embedding), DIT-03 (modulate)

**RMSNorm implementation** (lines 101–112):
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight
```
Key teaching points: upcasts to float32 for norm computation, no bias parameter, no mean centering.

**sinusoidal_embedding_1d implementation** (lines 68–72):
```python
def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)
```
Key teaching points: float64 precision internally, returns input dtype, `dim` must be even.

**modulate implementation** (lines 64–65):
```python
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)
```
Key teaching points: scale is additive offset (not multiplicative override), gate is separate and multiplies the residual branch (in GateModule).

**NB-01 verification cells to copy:**
```python
# RMSNorm shape assertion
B, S, dim = 1, 10, 1536
x = torch.randn(B, S, dim)
norm = RMSNorm(dim=dim)
out = norm(x)
assert out.shape == torch.Size([B, S, dim]), f"Expected {(B, S, dim)}, got {out.shape}"
print(f"RMSNorm output: {out.shape}")

# sinusoidal_embedding_1d shape assertion
freq_dim = 256
timesteps = torch.arange(50, dtype=torch.float32)
emb = sinusoidal_embedding_1d(freq_dim, timesteps)
assert emb.shape == torch.Size([50, freq_dim])
print(f"Sinusoidal embedding: {emb.shape}")

# modulate gate=0 identity
shift = torch.zeros(B, 1, dim)
scale = torch.zeros(B, 1, dim)
out = modulate(x, shift, scale)
assert torch.allclose(out, x), "scale=0, shift=0 must be identity"
print("gate=0: modulate is identity")
```

**Exercises:**
1. Swap RMSNorm for `nn.LayerNorm(dim)` — observe that LayerNorm adds a bias parameter and centers the mean.
2. Change `freq_dim` in sinusoidal embedding from 256 to 64 — verify `emb.shape[-1]` updates accordingly.
3. Set `scale = torch.ones(B, 1, dim)` in modulate — verify the output is `2*x + shift`.

---

### `Course/NB-02-qkv-projections-head-layout.ipynb` (notebook, transform)

**Analog:** `diffsynth/models/wan_video_dit.py`
**Covers:** DIT-04 (QKV projections), DIT-05 (multi-head layout conventions)

**QKV projection pattern** (lines 132–135 inside SelfAttention):
```python
self.q = nn.Linear(dim, dim)
self.k = nn.Linear(dim, dim)
self.v = nn.Linear(dim, dim)
self.o = nn.Linear(dim, dim)
```
No dimension expansion — all projections are `dim → dim`. Head splitting happens via einops.

**Multi-head split — dual convention** (lines 30–60, flash_attention):
```python
# Convention used by flash_attn_2/3 — sequence-first layout:
q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)  # [B, S, N, D]

# Convention used by PyTorch SDPA and SageAttn — head-first layout:
q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)  # [B, N, S, D]
```

**Inline shape annotation pattern (STD-02)** — copy this style exactly:
```python
B, S, dim, num_heads = 1, 20, 1536, 12
head_dim = dim // num_heads            # 128

x = torch.randn(B, S, dim)            # [B, S, dim]
q_proj = nn.Linear(dim, dim)
q = q_proj(x)                          # [B, S, dim]

q_bsnd = rearrange(q, "b s (n d) -> b s n d", n=num_heads)  # [B, S, N, D]
q_bnsd = rearrange(q, "b s (n d) -> b n s d", n=num_heads)  # [B, N, S, D]

assert q_bsnd.shape == torch.Size([B, S, num_heads, head_dim])
assert q_bnsd.shape == torch.Size([B, num_heads, S, head_dim])
print(f"(B,S,N,D): {q_bsnd.shape}")
print(f"(B,N,S,D): {q_bnsd.shape}")
```

**Exercises:**
1. Change `num_heads` from 12 to 8 — verify `head_dim` becomes 192 and shapes update correctly.
2. After splitting to (B,N,S,D), convert back to (B,S,N*D) using `rearrange(q_bnsd, "b n s d -> b s (n d)")` — verify it matches the original `q`.
3. Try `dim=1536, num_heads=7` — observe that `dim % num_heads != 0` raises an error from einops.

---

### `Course/NB-03-3d-rope.ipynb` (notebook, transform)

**Analog:** `diffsynth/models/wan_video_dit.py`
**Covers:** DIT-06 (3D RoPE frequency bands), DIT-07 (precompute_freqs_cis_3d)

**precompute_freqs_cis (1D base)** (lines 83–89):
```python
def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
```
Key: `torch.polar` requires float64 inputs; returns complex tensor of shape `[end, dim//2]`.

**precompute_freqs_cis_3d (3D split)** (lines 75–80):
```python
def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis
```

**Unequal band arithmetic (Pitfall 2 — must teach explicitly):**
```python
head_dim = 128
f_dim = head_dim - 2 * (head_dim // 3)  # 128 - 2*42 = 44  (temporal absorbs rounding)
h_dim = head_dim // 3                   # 42
w_dim = head_dim // 3                   # 42
assert f_dim + h_dim + w_dim == head_dim  # must sum to 128
```

**rope_apply (dtype handling)** (lines 92–98):
```python
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
```
Key: upcasts to float64, returns original dtype. Show `x.dtype` before and after to verify.

**3D frequency grid assembly** (lines 381–385 in WanModel.forward):
```python
F, H, W = 5, 4, 4
freqs = torch.cat([
    self.freqs[0][:F].view(F, 1, 1, -1).expand(F, H, W, -1),  # temporal
    self.freqs[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),  # height
    self.freqs[2][:W].view(1, 1, W, -1).expand(F, H, W, -1),  # width
], dim=-1).reshape(F * H * W, 1, -1)  # [seq_len, 1, head_dim//2]
assert freqs.shape == torch.Size([F * H * W, 1, head_dim // 2])
```

**Exercises:**
1. Change `head_dim` from 128 to 64 — verify `f_dim = 64 - 2*(64//3) = 22`, `h_dim = w_dim = 21`.
2. Change `theta` from 10000 to 1000 in `precompute_freqs_cis` — visualize how the frequency range changes.
3. In `rope_apply`, remove the `.to(torch.float64)` upcast — observe the `RuntimeError` about dtype.

---

### `Course/NB-04-self-cross-attention.ipynb` (notebook, request-response)

**Analog:** `diffsynth/models/wan_video_dit.py`
**Covers:** DIT-08 (SelfAttention with RoPE), DIT-09 (CrossAttention dual-stream)

**SelfAttention forward pass** (lines 125–148):
```python
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))   # RMSNorm on q before RoPE
        k = self.norm_k(self.k(x))   # RMSNorm on k before RoPE
        v = self.v(x)                 # v is NOT normalized
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)
```

**CrossAttention dual-stream logic** (lines 172–186):
```python
def forward(self, x: torch.Tensor, y: torch.Tensor):
    if self.has_image_input:
        img = y[:, :257]    # CLIP: 1 CLS token + 256 patch tokens
        ctx = y[:, 257:]    # T5 text tokens
    else:
        ctx = y
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(ctx))
    v = self.v(ctx)
    x = self.attn(q, k, v)         # main text cross-attention
    if self.has_image_input:
        k_img = self.norm_k_img(self.k_img(img))
        v_img = self.v_img(img)
        y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
        x = x + y                  # image cross-attention added to output
    return self.o(x)
```

**SelfAttention demo cell (simplified freqs for NB-04):**
```python
B, S, dim, num_heads = 1, 20, 1536, 12
head_dim = dim // num_heads          # 128

sa = SelfAttention(dim=dim, num_heads=num_heads)
x = torch.randn(B, S, dim)          # [B, S, dim]

# Simplified freqs: use temporal band only
# (Full 3D assembly shown in NB-03; this is a pedagogical shortcut)
f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
freqs = f_freqs[:S].unsqueeze(1)    # [S, 1, f_dim//2]

with torch.no_grad():
    out = sa(x, freqs)
assert out.shape == torch.Size([B, S, dim])
print(f"SelfAttention output: {out.shape}")
```

**CrossAttention demo cell:**
```python
B, S_vid, S_text, dim, num_heads = 1, 20, 30, 1536, 12

ca = CrossAttention(dim=dim, num_heads=num_heads, has_image_input=True)
x = torch.randn(B, S_vid, dim)         # video tokens — queries
y = torch.randn(B, 257 + S_text, dim)  # [CLIP(257) | text(S_text)] — keys/values

with torch.no_grad():
    out = ca(x, y)
assert out.shape == torch.Size([B, S_vid, dim])
print(f"CrossAttention output: {out.shape}")
# Internal split: img = y[:, :257], ctx = y[:, 257:]
```

**Exercises:**
1. Set `has_image_input=False` in CrossAttention — verify output shape is unchanged, confirm no `k_img`/`v_img` parameters exist.
2. Change `num_heads` from 12 to 6 in SelfAttention — verify output shape is unchanged but internal `head_dim` doubles to 256.
3. Remove `norm_q` and `norm_k` calls from SelfAttention forward (pass `self.q(x)` directly to `rope_apply`) — observe if output shape changes or if training instability warnings appear.

---

### `Course/NB-05-adaln-zero-modulation.ipynb` (notebook, event-driven)

**Analog:** `diffsynth/models/wan_video_dit.py`
**Covers:** DIT-10 (adaLN-Zero, gate=0 identity), DIT-11 (per-block modulation parameter)

**DiTBlock.modulation parameter** (line 212):
```python
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```
Shape: `[1, 6, dim]`. This is a **learned per-block additive offset** on the time projection. It is NOT zero-initialized (contrast with the original DiT paper, which zero-initializes the final linear layer).

**Six-parameter chunk from DiTBlock.forward** (lines 219–220):
```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
```

**GateModule (lines 190–195) — the gate=0 identity mechanism:**
```python
class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual  # gate=0 => output == x (identity)
```

**adaLN-Zero demo — six-parameter extraction:**
```python
dim = 1536
modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

# Zero time signal — isolates the modulation parameter's contribution
t_mod = torch.zeros(1, 6, dim)
combined = modulation + t_mod
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = combined.chunk(6, dim=1)
# Each chunk: [1, 1, dim]
for name, val in zip(
    ["shift_msa","scale_msa","gate_msa","shift_mlp","scale_mlp","gate_mlp"],
    [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
):
    print(f"  {name}: {val.shape}")
```

**gate=0 identity demonstration:**
```python
B, S, dim = 1, 10, 1536
x = torch.randn(B, S, dim)             # residual stream before block
residual = torch.randn(B, S, dim)       # output of self-attention branch

gate_zero = torch.zeros(B, 1, dim)
out_zero = x + gate_zero * residual
assert torch.allclose(out_zero, x), "gate=0 must be identity"
print("gate=0: block output == input (training starts from identity)")

gate_one = torch.ones(B, 1, dim)
out_one = x + gate_one * residual
assert not torch.allclose(out_one, x)
print("gate=1: residual branch fully incorporated")
```

**Pitfall 4 — must address in prose:** Wan2.1 uses `torch.randn(...) / dim**0.5` (small random), NOT zero-initialization. Show gate=0 identity as a constructed example, then separately explain that `modulation` is random-init. The zero-init motivation from the DiT paper is still valid pedagogically — use it as the "why" before showing the "what Wan2.1 actually does."

**Exercises:**
1. Change `t_mod = torch.zeros(...)` to `t_mod = torch.ones(...)` — observe how the six parameters shift away from the learned modulation baseline.
2. Set all six gate values to 0 in a DiTBlock — verify that `forward()` returns `x` unchanged (identity block behavior).
3. Increase the modulation scale divisor from `dim**0.5` to `dim` — compare the magnitude of the six parameters.

---

## Shared Patterns

### Import Setup Cell (STD-07)
**Source:** Verified against `diffsynth/models/wan_video_dit.py` (RESEARCH.md Pattern 2)
**Apply to:** ALL five notebooks — copy this cell verbatim as Cell 2

```python
import sys
import types
import importlib.util
import pathlib

# Adjust PROJECT_ROOT if notebook is not run from the Course/ directory
PROJECT_ROOT = pathlib.Path("..").resolve()  # Course/ is one level below project root

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

# Import all symbols used across notebooks
from diffsynth.models.wan_video_dit import (
    RMSNorm, sinusoidal_embedding_1d, modulate,
    precompute_freqs_cis, precompute_freqs_cis_3d,
    rope_apply, flash_attention, SelfAttention, CrossAttention,
    GateModule, DiTBlock,
)
import torch
import torch.nn as nn
from einops import rearrange
print("Setup complete.")
```

**Why importlib (not `import diffsynth`):** `diffsynth/__init__.py` triggers a full import chain requiring `modelscope`, which is not installed. Direct module loading via `importlib.util.spec_from_file_location` bypasses this entirely. Verified working in project environment.

### Shape Assertion Pattern (STD-04)
**Source:** Research Pattern 3
**Apply to:** Every code cell that produces a tensor output

```python
assert output.shape == torch.Size([B, S, dim]), \
    f"Expected ({B}, {S}, {dim}), got {output.shape}"
print(f"Shape OK: {output.shape}")
```

### Inline Shape Annotation Pattern (STD-02)
**Source:** `diffsynth/models/wan_video_dit.py` lines 30–60 (flash_attention)
**Apply to:** Every `rearrange` call, every `nn.Linear` call, every tensor operation with non-obvious output shape

```python
q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)  # [B, S, N, D]
q = rearrange(q, "b s n d -> b n s d")                  # [B, N, S, D]
```
The comment MUST show the output shape. For `nn.Linear`:
```python
q = self.q(x)    # [B, S, dim] -> [B, S, dim]
```

### Source Citation Pattern (STD-06)
**Source:** RESEARCH.md Source Line Reference Table
**Apply to:** First code cell introducing each symbol; also in markdown prose before the cell

Standard citation comment format:
```python
# Source: diffsynth/models/wan_video_dit.py, line <N>
```

Full reference table (copy into each notebook as needed):
| Symbol | File | Line |
|--------|------|------|
| `flash_attention` | `diffsynth/models/wan_video_dit.py` | 28 |
| `modulate` | `diffsynth/models/wan_video_dit.py` | 64 |
| `sinusoidal_embedding_1d` | `diffsynth/models/wan_video_dit.py` | 68 |
| `precompute_freqs_cis_3d` | `diffsynth/models/wan_video_dit.py` | 75 |
| `precompute_freqs_cis` | `diffsynth/models/wan_video_dit.py` | 83 |
| `rope_apply` | `diffsynth/models/wan_video_dit.py` | 92 |
| `RMSNorm` | `diffsynth/models/wan_video_dit.py` | 101 |
| `SelfAttention` | `diffsynth/models/wan_video_dit.py` | 125 |
| `CrossAttention` | `diffsynth/models/wan_video_dit.py` | 151 |
| `GateModule` | `diffsynth/models/wan_video_dit.py` | 190 |
| `DiTBlock.modulation` | `diffsynth/models/wan_video_dit.py` | 212 |

### Dummy Tensor Dimensions (STD-03)
**Source:** Verified via execution — RESEARCH.md "All DiT primitives run on CPU in well under 5 seconds"
**Apply to:** All code cells creating input tensors

Use these standard dimensions for CPU execution under 5 seconds:
```python
B = 1          # batch size
S = 20         # sequence length (50 max for NB-04 timing budget)
dim = 1536     # model dimension (matches production Wan2.1-1.3B)
num_heads = 12
head_dim = 128  # dim // num_heads
freq_dim = 256  # for sinusoidal embedding
```
Never use `x.cuda()` or `.to("cuda")`. All cells must run on CPU.

### Prerequisite Header Pattern (STD-01)
**Source:** D-01 (locked decision)
**Apply to:** Cell 1 (Markdown) of every notebook

```markdown
# NB-0X: [Title]

## Learning Objectives
- [Objective 1 — concrete and measurable]
- [Objective 2]
- [Objective 3]

## Prerequisites
- **Prior notebooks:** NB-0X (or "None" for NB-01)
- **Assumed concepts:** [list — e.g., transformer attention, PyTorch nn.Module basics]

## Concept Map
- [Symbol] → [where it appears next, e.g., "used in DiT blocks (NB-06)"]
- [Symbol] → [forward reference]
```

---

## No Analog Found

All notebooks in this phase are entirely new with no prior notebooks to reference. The source-code analog (`diffsynth/models/wan_video_dit.py`) provides the implementation patterns; the RESEARCH.md provides all verified code examples. Planner should use the patterns above and the RESEARCH.md Code Examples section directly.

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` | notebook | transform | No prior notebooks exist in this repo |
| `Course/NB-02-qkv-projections-head-layout.ipynb` | notebook | transform | No prior notebooks exist in this repo |
| `Course/NB-03-3d-rope.ipynb` | notebook | transform | No prior notebooks exist in this repo |
| `Course/NB-04-self-cross-attention.ipynb` | notebook | request-response | No prior notebooks exist in this repo |
| `Course/NB-05-adaln-zero-modulation.ipynb` | notebook | event-driven | No prior notebooks exist in this repo |

---

## Anti-Patterns (Do Not Use)

Extracted from RESEARCH.md — these are verified failure modes:

| Anti-Pattern | Why Banned | Correct Alternative |
|---|---|---|
| `import diffsynth` | Triggers `modelscope` import failure | Use importlib setup cell (Shared Pattern) |
| `class MyRMSNorm(nn.Module): ...` inline | Violates STD-07 (real classes required) | `from diffsynth.models.wan_video_dit import RMSNorm` |
| `x.cuda()` or `.to("cuda")` | Notebooks must run on CPU | No device move; torch defaults to CPU |
| `torch.load("wan_video.pth")` | Requires model download; breaks portability | Random `torch.randn(...)` dummy tensors only |
| Code cell before markdown prose | Violates STD-05 | Always markdown → code → assert ordering |
| Omitting shape comment on rearrange | Violates STD-02 | Add `# [B, S, N, D]` comment on every rearrange |
| Presenting 3D bands as equal thirds | Misleads reader (f_dim=44, not 42) | Show `f_dim = dim - 2*(dim//3)` arithmetic explicitly |
| "gates are zero-initialized" | Inaccurate for Wan2.1 (`torch.randn/dim**0.5`) | Demonstrate gate=0 identity by manual construction; explain modulation init separately |

---

## Metadata

**Analog search scope:** `diffsynth/models/` (primary), `Course/` (empty — no existing notebooks)
**Files scanned:** `wan_video_dit.py` (400 lines read), `general_modules.py` (60 lines read)
**Source line numbers:** All verified against actual file via grep in RESEARCH.md
**Pattern extraction date:** 2026-04-24
