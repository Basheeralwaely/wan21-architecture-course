# Phase 4: System Integration - Pattern Map

**Mapped:** 2026-04-24
**Files analyzed:** 1 new file (NB-12)
**Analogs found:** 11 / 11 (all prior notebooks serve as analogs; NB-01, NB-06, NB-08, NB-11 are primary)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `Course/NB-12-pipeline-system-integration.ipynb` | notebook (capstone) | transform (encode → denoise → decode) | `Course/NB-08-wanmodel-forward.ipynb` | exact — same multi-component orchestration pattern |

Secondary analogs used for specific sections:

| Section | Analog Notebook | Match |
|---------|----------------|-------|
| Title / Learning Objectives / Prerequisites | `NB-01` Cell 0 (markdown) | template |
| Setup cell (importlib, stubs) | `NB-01` Cell 1, `NB-08` Cell 2, `NB-11` Cell 1 | exact — same sys.path workaround |
| Parameter count breakdown table | `NB-06` Cells 12–13 | exact — same `sum(p.numel() ...)` loop + table |
| Component-by-component demo | `NB-08` Cells 4–6 | exact — reduced config, assert shape, print |
| ASCII architecture diagram | `NB-08` Cell 1 data flow box | style match |
| Round-trip encode/decode demo | `NB-11` Cell 7 | exact — same pattern |
| Exercises section | `NB-01` Cell final, `NB-06` Cell final | template |

---

## Pattern Assignments

### `Course/NB-12-pipeline-system-integration.ipynb` (capstone notebook)

**Primary analog:** `Course/NB-08-wanmodel-forward.ipynb`
**Secondary analogs:** NB-01, NB-06, NB-11

---

#### PATTERN 1 — Title / Learning Objectives / Prerequisites (Cell 0 markdown)

**Analog:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` Cell `712ac433` (first cell)

```markdown
# NB-01: RMSNorm, Sinusoidal Embeddings, and the Modulate Function

## Learning Objectives
- Understand why ...
- Trace how ...
- See how ...

## Prerequisites
- **Prior notebooks:** None (this is the first notebook in the series)
- **Assumed concepts:** PyTorch nn.Module, ...

## Concept Map
- RMSNorm → used in ... (NB-04) and ... (NB-06)
- sinusoidal_embedding_1d → used in ... (NB-08)
- modulate → used in ... (NB-05) and ... (NB-06)
```

**Apply to NB-12 Cell 0 as:**
- Title: `# NB-12: WanVideoPipeline — Full System Integration`
- Three learning objectives covering: (1) pipeline composition, (2) tensor flow from input to output, (3) parameter count across all components
- Prerequisites listing NB-01 through NB-11
- Concept Map mapping each notebook to the pipeline stage it covers

---

#### PATTERN 2 — Setup Cell (importlib stubs, project root discovery)

**Analog:** `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-02` (lines as extracted)

```python
import sys
import types
import importlib.util
import pathlib

# Find project root: walk up from Course/ to find the directory containing diffsynth/
_here = pathlib.Path().resolve()
PROJECT_ROOT = None
_candidate = _here
for _ in range(10):  # search up to 10 levels
    if (_candidate / "diffsynth" / "models" / "wan_video_dit.py").exists():
        PROJECT_ROOT = _candidate
        break
    _parent = _candidate.parent
    if _parent == _candidate:
        break
    _candidate = _parent
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

from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
import torch

print("Setup complete.")
```

**NB-12 extension:** The setup cell must also load `wan_video_vae.py`, `wan_video_text_encoder.py`, `wan_video_image_encoder.py`, and `flow_match.py` via the same `importlib.util.spec_from_file_location` pattern. Additional stubs are required for `wan_video_image_encoder.py` — it imports from `tqdm`, `transformers`, and optional CLIP libraries that may or may not be present. Follow the same stub approach as the `_cam_stub` pattern above.

**NB-11 VAE variant** (`Course/NB-11-vae-decoder-patchify.ipynb` Cell `nb11-cell-01`) shows the additional tqdm stub needed when loading VAE:

```python
# Stub tqdm to suppress progress bars in notebook
_tqdm_stub = types.ModuleType('tqdm')
_tqdm_stub.tqdm = lambda x, **kw: x
sys.modules['tqdm'] = _tqdm_stub
```

---

#### PATTERN 3 — ASCII Architecture Diagram (Data Flow, D-01/D-02)

**Analog:** `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-01` data flow box

```markdown
## WanModel Data Flow

```
WanModel.forward -- 8-Step Data Flow
=====================================
  Input: x [B, 48, F, H, W]  (48 = 16 noise + 16 ctrl + 16 ref)
         ...
         |
         +-- Step 1: time_embedding   [B] -> sinusoidal -> [B, 256] -> Linear+SiLU+Linear -> t [B, 1536]
         ...
  Output: x [B, 16, F, H, W]  (denoised output latents)
```
```

**NB-12 must use the full pipeline diagram from RESEARCH.md** (the one showing T5/CLIP/VAE encode → 48-ch concat → DiT denoise → VAE decode), not just WanModel's internal flow. Each component box should include back-references:

```
WanTextEncoder (T5/UMT5-XXL, 24 layers)    [← NB-01 RMSNorm, NB-02 Attention]
WanImageEncoder (CLIP ViT-H/14)             [← NB-02, NB-04]
WanVideoVAE.encode                          [← NB-10]
48-Channel Concatenation                    [← NB-08]
WanModel (DiT, 30 blocks)                   [← NB-06, NB-07, NB-08]
FlowMatchScheduler (50 steps)               [← new in NB-12]
WanVideoVAE.decode                          [← NB-11]
```

---

#### PATTERN 4 — Component-by-Component Demo (D-04)

**Analog:** `Course/NB-08-wanmodel-forward.ipynb` Cells `nb08-cell-04` and `nb08-cell-06`

Cell `nb08-cell-04` (48-ch concat demo style):
```python
# Source: model_architecture.md + diffsynth/models/wan_video_dit.py WanModel.forward
B, F, H, W = 1, 4, 8, 8  # small spatial dims for CPU demo

noise_latent   = torch.randn(B, 16, F, H, W)
control_latent = torch.randn(B, 16, F, H, W)
ref_latent     = torch.randn(B, 16, F, H, W)

x_48 = torch.cat([noise_latent, control_latent, ref_latent], dim=1)
assert x_48.shape == torch.Size([B, 48, F, H, W]), \
    f"Expected ({B}, 48, {F}, {H}, {W}), got {x_48.shape}"

print(f"noise:    {noise_latent.shape}")
print(f"control:  {control_latent.shape}")
print(f"ref:      {ref_latent.shape}")
print(f"48-ch x:  {x_48.shape}  <- WanModel.in_dim=48")
```

Cell `nb08-cell-06` (WanModel reduced-config instantiation style):
```python
# DEMO CONFIG (3 layers, STD-03 compliant -- verified at 0.042s on CPU)
model = WanModel(
    dim=1536, in_dim=48, ffn_dim=8960, out_dim=16,
    text_dim=4096, freq_dim=256, eps=1e-6,
    patch_size=(1, 2, 2), num_heads=12,
    num_layers=3,       # REDUCED from 30 for CPU demo
    has_image_input=False,
)

# PRODUCTION CONFIG (annotation only -- do not run on CPU)
# model = WanModel(..., num_layers=30, has_image_input=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Demo model (3 layers): {total_params:,} parameters")
```

**NB-12 must apply this pattern to each of four components:**

T5 demo (reduced config per RESEARCH.md Pitfall 2):
```python
# Source: diffsynth/models/wan_video_text_encoder.py, class WanTextEncoder line 212
# DEMO CONFIG: vocab=1000, num_layers=2 (production: vocab=256384, num_layers=24)
# CRITICAL: Full T5 takes ~35s to instantiate on CPU (1.05B embedding params)
text_encoder = WanTextEncoder(
    vocab=1000,      # reduced from 256,384 for speed
    dim=4096,
    dim_attn=4096,
    dim_ffn=10240,
    num_heads=64,
    num_layers=2,    # reduced from 24
    num_buckets=32,
)
dummy_ids  = torch.zeros(1, 10, dtype=torch.long)
dummy_mask = torch.ones(1, 10)
context = text_encoder(dummy_ids, dummy_mask)    # -> [B, L, 4096]
assert context.shape == (1, 10, 4096)
print(f"T5 output shape: {context.shape}")
```

CLIP demo (full config — 2.9s total, within STD-03):
```python
# Source: diffsynth/models/wan_video_image_encoder.py, class WanImageEncoder line 852
# Full CLIP ViT-H/14 -- no reduced version needed (2.4s init + 0.45s forward = 2.9s < 5s)
image_encoder = WanImageEncoder()
dummy_img = torch.randn(1, 3, 224, 224)
clip_feature = image_encoder.encode_image([dummy_img])   # -> [B, 257, 1280]
assert clip_feature.shape == (1, 257, 1280)
print(f"CLIP output shape: {clip_feature.shape}")
```

VAE encode/decode demo (from NB-11 Cell `nb11-cell-07` pattern):
```python
# Source: diffsynth/models/wan_video_vae.py, class WanVideoVAE line 1058
vae = WanVideoVAE()
dummy_video = torch.randn(1, 3, 5, 32, 32)   # [B, C, F, H, W]
with torch.no_grad():
    latents = vae.encode(dummy_video, device='cpu')   # -> [B, 16, 2, 4, 4]
    decoded = vae.decode(latents, device='cpu')       # -> [B, 3, 5, 32, 32]
assert latents.shape[1] == 16
assert decoded.shape == dummy_video.shape
print(f"VAE latents: {latents.shape}")
print(f"Decoded:     {decoded.shape}")
```

DiT demo (same as NB-08 Cell `nb08-cell-06`):
```python
dit = WanModel(
    dim=1536, in_dim=16, ffn_dim=8960, out_dim=16,
    text_dim=4096, freq_dim=256, eps=1e-6,
    patch_size=(1,2,2), num_heads=12,
    num_layers=3,          # 3 for demo; production = 30
    has_image_input=False,
)
```

---

#### PATTERN 5 — 48-Channel Concatenation Demo (SYS-02)

**Analog:** `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-04`

```python
# Source: diffsynth/pipelines/wan_video.py, model_fn_wan_video
# CRITICAL: WanModel.forward takes x=[B,16,...] and y=[B,32,...] SEPARATELY.
# The model does the cat internally when has_image_input=True.
# Do NOT pass a pre-concatenated 48-ch tensor.
#
# Source: wan_video_dit.py line 359 -- WanModel.forward signature:
#   def forward(self, x, timestep, context, clip_feature=None, y=None, ...)
#   When has_image_input=True: x = torch.cat([x, y], dim=1)  inside forward
#
# The 48-channel assembly (for reference/pipeline discussion):
B, C, T, H, W = 1, 16, 4, 8, 8
noise_latents   = torch.randn(B, 16, T, H, W)
control_latents = torch.randn(B, 16, T, H, W)
ref_embedding   = torch.randn(B, 16, T, H, W)

# model_fn_wan_video assembles y = cat(control, ref_emb):
y = torch.cat([control_latents, ref_embedding], dim=1)   # [B, 32, T, H, W]
# Then WanModel.forward does: x = cat([noise_latents, y], dim=1) -> [B, 48, T, H, W]
assert y.shape[1] == 32
print(f"noise x: {noise_latents.shape}")
print(f"y (control+ref): {y.shape}")
print(f"Combined in-model: [B, {noise_latents.shape[1]+y.shape[1]}, T, H, W]")
```

---

#### PATTERN 6 — FlowMatchScheduler Denoising Loop (D-03)

**Analog:** No prior notebook covers FlowMatchScheduler. Patterns sourced directly from `diffsynth/diffusion/flow_match.py` lines 5–175.

FlowMatchScheduler instantiation and timestep setup (lines 5–39, 137–148):
```python
# Source: diffsynth/diffusion/flow_match.py, lines 5-39, 137-148
from diffsynth.diffusion.flow_match import FlowMatchScheduler

scheduler = FlowMatchScheduler("Wan")
scheduler.set_timesteps(
    num_inference_steps=3,      # 3 for demo; production uses 50
    denoising_strength=1.0,     # 1.0 = full denoising from noise
    shift=5.0,                  # Wan default
)
# scheduler.sigmas:    [1.000, ..., 0.093]   (3 values, descending)
# scheduler.timesteps: [1000.0, ..., 92.6]   (3 values, = sigmas * 1000)
print(f"timesteps: {scheduler.timesteps}")
print(f"sigmas:    {scheduler.sigmas}")
```

Step formula (lines 149–159):
```python
# Source: diffsynth/diffusion/flow_match.py, lines 149-159
# step() implements: prev_sample = sample + model_output * (sigma_next - sigma_current)
# Since sigma_next < sigma_current, delta is NEGATIVE -> moves toward clean image
# model_output is velocity (direction from noise to data), NOT noise

def step(self, model_output, timestep, sample, to_final=False, **kwargs):
    timestep_id = torch.argmin((self.timesteps - timestep).abs())
    sigma  = self.sigmas[timestep_id]
    sigma_ = 0 if to_final else self.sigmas[timestep_id + 1]
    prev_sample = sample + model_output * (sigma_ - sigma)  # delta is negative
    return prev_sample
```

Conceptual denoising loop pattern (adapted from `wan_video.py` lines 290–314):
```python
# Source: diffsynth/pipelines/wan_video.py, lines 290-314
# Demo: 3 steps, base T2V DiT (no image input), simplified context
scheduler.set_timesteps(3)
latents = torch.randn(1, 16, T_lat, H_lat, W_lat)

for i, timestep in enumerate(scheduler.timesteps):
    t = timestep.unsqueeze(0)
    # Positive prediction
    noise_pred_posi = dit(latents, t, context)
    # Negative prediction (empty/zero context)
    noise_pred_nega = dit(latents, t, context_neg)
    # CFG combination (lines 301-309)
    cfg_scale = 7.5
    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
    # Update latents (lines 149-159)
    latents = scheduler.step(noise_pred, timestep, latents)
```

---

#### PATTERN 7 — Parameter Count Table (D-05, SYS-03)

**Analog:** `Course/NB-06-dit-block.ipynb` Cells `nb06-cell-12` and `nb06-cell-13`

Per-component loop pattern from NB-06 Cell `nb06-cell-12`:
```python
# Source: NB-06, cells nb06-cell-12 and nb06-cell-13
# Per-block LoRA target parameter counts
lora_targets = {
    "self_attn.q": sum(p.numel() for p in block.self_attn.q.parameters()),
    ...
}
block_total = sum(p.numel() for p in block.parameters())
lora_total_per_block = sum(lora_targets.values())

print("Per-block LoRA target parameter counts:")
for name, count in lora_targets.items():
    pct = count / block_total * 100
    print(f"  {name:<20}: {count:>12,}  ({pct:.1f}%)")
```

Full-model scaling pattern from NB-06 Cell `nb06-cell-13`:
```python
# Source: NB-06, cell nb06-cell-13
num_layers = 30
print(f"Full-model LoRA scaling ({num_layers} DiT blocks):")
print(f"  Per-block LoRA params:  {lora_total_per_block:,}")
print(f"  Number of blocks:       {num_layers}")
print(f"  Total LoRA params:      {lora_total_per_block * num_layers:,}")
```

**NB-12 must adapt this to a cross-component table using live demo counts + production extrapolation:**
```python
# Live computation for demo models (reduced configs)
components_demo = {
    "DiT (3-block demo, in_dim=16)": dit_demo,
    "VAE":                            vae,
    "T5 (2-layer demo)":              text_encoder_demo,
    "CLIP":                           image_encoder,
}
for name, model in components_demo.items():
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name:<35}: {n:>15,}")

# Production parameter table (verified values -- corrects model_architecture.md's "~780M" figure)
# Source: RESEARCH.md "Verified Parameter Counts" section (live-computed 2026-04-24)
production_counts = {
    "DiT (Fun-Control, in_dim=48, 30 layers)": 1_564_602_176,
    "VAE":                                      126_892_531,
    "T5 Text Encoder (UMT5-XXL, 24 layers)":   5_680_910_336,
    "CLIP Image Encoder (ViT-H/14)":            632_076_801,
}
total = sum(production_counts.values())  # 8,004,481,844

print("\nProduction parameter counts (measured via live instantiation):")
for name, count in production_counts.items():
    pct = count / total * 100
    print(f"  {name:<45}: {count:>15,}  ({pct:.1f}%)")
print(f"  {'TOTAL':<45}: {total:>15,}")

# NOTE: model_architecture.md states "~780M DiT params" -- THIS IS WRONG.
# Measured: 1.56B (Fun-Control, in_dim=48) or 1.42B (base T2V, in_dim=16).
# NB-12 must use the measured values above and document the discrepancy.
```

---

#### PATTERN 8 — Summary + Key Takeaways (final markdown cell)

**Analog:** All prior notebooks; canonical template from `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-15`

```markdown
## Summary

### Key Takeaways
- **[Topic 1]**: ...
- **[Topic 2]**: ...
- **[Topic 3]**: ...

### Source References
| Symbol | Location |
|--------|----------|
| `WanModel.__init__` | `diffsynth/models/wan_video_dit.py`, line 274 |
| `WanModel.forward`  | `diffsynth/models/wan_video_dit.py`, line 359 |
| ...
```

---

#### PATTERN 9 — Exercises (final cell)

**Analog:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` last cell (`1e445812`), `Course/NB-06-dit-block.ipynb` Cell `nb06-cell-15`

```markdown
## Exercises

### Exercise 1 — [Name]
[Setup: start from an existing cell]. [Action: change one thing]. [Verify: assert or print check]. [Question: what do you observe?]

### Exercise 2 — [Name]
[Same structure]

### Exercise 3 — [Name]
[Same structure]
```

---

## Shared Patterns

### Import / Module Loading (applies to ALL setup cells in this codebase)

**Source:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` Cell `c07635b0` and `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-02`

The `diffsynth/__init__.py` triggers a pandas/numpy version conflict on this machine. All notebooks bypass it using `importlib.util.spec_from_file_location`. The pattern is:

```python
import sys, types, importlib.util, pathlib

# 1. Walk up to find PROJECT_ROOT
_here = pathlib.Path().resolve()
PROJECT_ROOT = None
_candidate = _here
for _ in range(10):
    if (_candidate / "diffsynth" / "models" / "wan_video_dit.py").exists():
        PROJECT_ROOT = _candidate
        break
    _parent = _candidate.parent
    if _parent == _candidate:
        break
    _candidate = _parent

# 2. Stub modules that cause import errors
_cam_stub = types.ModuleType("diffsynth.models.wan_video_camera_controller")
_cam_stub.SimpleAdapter = type("SimpleAdapter", (), {"__init__": lambda s, *a, **kw: None})
sys.modules["diffsynth"] = types.ModuleType("diffsynth")
sys.modules["diffsynth.models"] = types.ModuleType("diffsynth.models")
sys.modules["diffsynth.models.wan_video_camera_controller"] = _cam_stub

# 3. Load each module file directly
_path = PROJECT_ROOT / "diffsynth" / "models" / "wan_video_dit.py"
_spec = importlib.util.spec_from_file_location("diffsynth.models.wan_video_dit", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["diffsynth.models.wan_video_dit"] = _mod
_spec.loader.exec_module(_mod)
```

**Apply to:** Every notebook setup cell in `Course/`.

---

### Reduced-Config Demo Pattern (applies to all component demo cells)

**Source:** `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-06`, `Course/NB-06-dit-block.ipynb` Cell `nb06-cell-04`

Every demo cell follows:
1. Comment header: `# DEMO CONFIG (...reduced param..., verified ...s on CPU)`
2. Instantiate with reduced config
3. `# PRODUCTION CONFIG (annotation only -- do not run on CPU)` as commented code block
4. `total_params = sum(p.numel() for p in model.parameters())`
5. Print demo count; extrapolate or print production count as annotation

**Apply to:** All four component demo cells (T5, CLIP, VAE, DiT) in NB-12.

---

### Shape Assertion Pattern (applies to all forward-pass cells)

**Source:** `Course/NB-08-wanmodel-forward.ipynb` Cell `nb08-cell-09`, `Course/NB-11-vae-decoder-patchify.ipynb` Cell `nb11-cell-07`

```python
assert out.shape == torch.Size([B, C, F, H, W]), \
    f"Expected ({B}, {C}, {F}, {H}, {W}), got {out.shape}"
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
```

**Apply to:** Every code cell that runs a forward pass through a model component.

---

### Source-Reference Comment Header (applies to all code cells)

**Source:** All notebooks — consistent across NB-01 through NB-11

```python
# Source: diffsynth/models/wan_video_dit.py, lines 359-416
# <brief description of what the cell demonstrates>
```

**Apply to:** First comment in every code cell in NB-12.

---

## No Analog Found

No files in this phase lack analogs. NB-12's content is entirely synthesis of existing patterns from NB-01 through NB-11. The only genuinely new section is the FlowMatchScheduler denoising loop — but its source code is directly available in `diffsynth/diffusion/flow_match.py` lines 5–175, which was read and excerpted in Pattern 6 above.

| Section | Status |
|---------|--------|
| FlowMatchScheduler usage | No prior notebook — use RESEARCH.md Pattern 3 + flow_match.py lines 149-159 directly |
| CFG explanation | No prior notebook — use RESEARCH.md Pattern 5 + wan_video.py lines 301-309 |
| WanVideoPipeline unit architecture prose | No prior notebook — read-only reference to wan_video.py lines 55-82 |

---

## Anti-Pattern Registry (from RESEARCH.md)

These must be called out explicitly in planner actions to avoid:

| Anti-Pattern | File | Correct Pattern |
|--------------|------|-----------------|
| Pass pre-cat `x=[B,48,...]` to `WanModel.forward` | `wan_video_dit.py:359` | Pass `x=[B,16,...]` + `y=[B,32,...]` separately |
| Use `model_architecture.md`'s "~780M DiT params" | `model_architecture.md` | Use RESEARCH.md measured values: 1.56B (Fun-Control) |
| Instantiate full `WanTextEncoder()` in a demo cell | T5 demo cell | Use `WanTextEncoder(vocab=1000, num_layers=2)` |
| Use `F//4` for VAE temporal compression | VAE shape trace | Use `(F-1)//4 + 1` |
| Explain denoising as "noise is added" | denoising loop cell | Flow matching predicts velocity; `sigma_next < sigma_current` so delta is negative |

---

## Metadata

**Analog search scope:** `Course/` (all 11 notebooks), `diffsynth/diffusion/flow_match.py`, `diffsynth/models/wan_video_text_encoder.py`, `diffsynth/models/wan_video_image_encoder.py`
**Files scanned:** 11 notebooks + 4 source files
**Pattern extraction date:** 2026-04-24
