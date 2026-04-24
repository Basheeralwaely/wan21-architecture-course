# Wan2.1-Fun-V1.1-1.3B-Control Model Architecture

All tensor shapes assume: **512x512 resolution, 81 frames, batch_size=1**

---

## Full Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WanVideoPipeline                                     │
│                                                                             │
│  ┌───────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │  Prompt   │   │ Ref Image    │   │Control Video│   │   Random Noise   │  │
│  │ (string)  │   │[3,512,512]   │   │[81,3,512,   │   │[1,16,21,64,64]   │  │
│  │           │   │              │   │     512]    │   │                  │  │
│  └─────┬─────┘   └──┬───────┬───┘   └──────┬──────┘   └───────┬──────────┘  │
│        │            │       │              │                  │             │
│        ▼           ▼       ▼              ▼                 │             │
│  ┌──────────┐ ┌─────────┐┌──────┐  ┌────────────┐             │             │
│  │T5 Text   │ │  CLIP   ││ VAE  │  │    VAE     │             │             │
│  │Encoder   │ │Encoder  ││Encode│  │   Encode   │             │             │
│  └────┬─────┘ └────┬────┘└──┬───┘  └──────┬─────┘             │             │
│       │            │        │             │                   │             │
│       ▼            ▼       ▼            ▼                   ▼             │
│  context      clip_feat  ref_lat       ctrl_lat              noise          │
│[B,L,4096] [B,257,1280] [B,16,1,64,64] [B,16,21,64,64]   [B,16,21,64,64]     │
│       │            │        │             │                   │             │
│       │            │        │             │                   │             │
│       │            │        │      ┌──────┘                   │             │
│       │            │        │      │  cat(noise,ctrl,ref)     │             │
│       │            │        │      │      ┌───────────────────┘             │
│       │            │        │      ▼      ▼                                │
│       │            │        │  ┌──────────────────┐                         │
│       └────────────┼────────┼─▶│   DiT (30 blocks)│ ──── Denoise Loop ───   │
│                    │        │  │   x 50 timesteps │      (50 steps)         │
│                    └────────┼─▶│                  │                         │
│                             └─▶│  [B,48,21,64,64] │                         │
│                                └────────┬─────────┘                         │
│                                         │                                   │
│                                         ▼                                   │
│                                   [B,16,21,64,64]                           │
│                                         │                                   │
│                                         ▼                                   │
│                                  ┌─────────────┐                            │
│                                  │  VAE Decode │                            │
│                                  └──────┬──────┘                            │
│                                         ▼                                   │
│                                  [B,3,81,512,512]                           │
│                                    Output Video                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Latent space dimensions for 512x512, 81 frames: `T_lat=21, H_lat=64, W_lat=64`
(spatial /8, temporal /4)

---

## DiT (Diffusion Transformer) — 30 Blocks, ~780M params

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DiT Model                                  │
│   dim=1536, heads=12, head_dim=128, ffn_dim=8960, layers=30             │
│                                                                         │
│  INPUTS:                                                                │
│  ┌──────────────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────┐   │
│  │ x (latents+ctrl) │ │ timestep  │ │  context   │ │  clip_feature  │   │
│  │[B,48,21,64,64]   │ │  [B]      │ │[B,L,4096]  │ │[B,257,1280]    │   │
│  └────────┬─────────┘ └─────┬─────┘ └─────┬──────┘ └───────┬────────┘   │
│           │                 │             │                │            │
│           ▼                 │            ▼                ▼            │
│  ┌────────────────┐         │     ┌──────────────┐  ┌──────────────┐    │
│  │ Patch Embed    │         │     │ Text Embed   │  │ CLIP MLP     │    │
│  │ Conv3d(48→1536 │        │     │ Lin(4096→     │   │ Lin(1280→     │    │
│  │  k=(1,2,2),    │         │     │  1536)→GELU │   │  1536)       │   │
│  │  s=(1,2,2))    │         │     │ →Lin(1536→   │  │              │    │
│  └────────┬───────┘         │     │  1536)       │  └──────┬───────┘    │
│           │                 │     └──────┬───────┘         │            │
│           ▼                │            │                  │           │
│  [B, 21·32·32, 1536]       │    [B,L,1536]          [B,257,1536]      │
│  = [B, 21504, 1536]        │            │                  │           │
│           │                │           └──────┬───────────┘           │
│           │                │                   │                       │
│  ┌────────┴────────┐       │            context_cat                    │
│  │ + ref_conv      │       │           [B, L+257, 1536]               │
│  │ ref_lat →         │       │                   │                       │
│  │ Conv2d(16→1536, │      ▼                   │                       │
│  │  k=2,s=2)       │  ┌────────────┐           │                       │
│  │ [B,1024,1536]   │  │ Time Embed │           │                       │
│  │ prepend to seq  │  │ sin→Lin→   │           │                       │
│  └────────┬────────┘  │ SiLU→Lin   │          │                       │
│           │           │ [B,1536]    │          │                       │
│  [B, 22528, 1536]     │      │      │          │                       │
│  (21504+1024)         │      ▼      │          │                       │
│           │           │ ┌────────┐  │          │                       │
│           │           │ │TimProj │  │          │                       │
│           │           │ │Lin(1536│  │          │                       │
│           │           │ │ →9216) │  │          │                       │
│           │           │ └───┬────┘  │          │                       │
│           │           │     ▼       │          │                       │
│           │           │ [B,6,1536]  │          │                       │
│           │           │ (shift_sa,  │          │                       │
│           │           │  scale_sa,  │          │                       │
│           │           │  gate_sa,   │          │                       │
│           │           │  shift_ff,  │          │                       │
│           │           │  scale_ff,  │          │                       │
│           │           │  gate_ff)   │          │                       │
│           │           └─────┬───────┘          │                       │
│           │                 │                  │                       │
│           ▼                 ▼                 ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                  ×30 DiT Blocks (see below)                 │       │
│  │  x: [B, 22528, 1536]                                        │       │
│  │  context: [B, L+257, 1536]                                  │       │
│  │  t_mod: [B, 6, 1536]                                        │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│                    [B, 22528, 1536]                                    │
│                      strip ref tokens → [B, 21504, 1536]               │
│                             │                                          │
│                             ▼                                          │
│                    ┌──────────────┐                                    │
│                    │  Output Head │                                    │
│                    │ Lin(1536→64) │   64 = 16 × 1 × 2 × 2             │
│                    └──────┬───────┘                                    │
│                           ▼                                            │
│                  Unpatchify → [B, 16, 21, 64, 64]                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Single DiT Block (×30)

```
┌────────────────────────────────────────────────────────────────┐
│                     DiT Block                                  │
│                                                                │
│  x [B, 22528, 1536]     context [B, L+257, 1536]               │
│  t_mod [B, 6, 1536]     freqs (3D ROPE)                        │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SELF-ATTENTION                                          │   │
│  │                                                         │   │
│  │  x ──→ RMSNorm(1536) ──→ modulate(scale_sa, shift_sa)  │   │
│  │         │                                               │   │
│  │         ├──→ q: Linear(1536→1536) ──→ RMSNorm          │   │
│  │         ├──→ k: Linear(1536→1536) ──→ RMSNorm          │   │
│  │         └──→ v: Linear(1536→1536)                      │   │
│  │                                                         │   │
│  │  Reshape: [B, 22528, 1536] → [B, 12, 22528, 128]        │   │
│  │  Apply 3D ROPE to q, k                                  │   │
│  │  Flash Attention → [B, 12, 22528, 128]                  │   │
│  │  Reshape → [B, 22528, 1536]                             │   │
│  │                                                         │   │
│  │  o: Linear(1536→1536) × gate_sa                        │   │
│  │  Residual: x = x + output                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                        │
│       ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ CROSS-ATTENTION                                        │   │
│  │                                                        │   │
│  │  x ──→ q: Linear(1536→1536) → RMSNorm                 │   │
│  │                                                         │   │
│  │  context ──→ k: Linear(1536→1536) → RMSNorm           │   │
│  │          └──→ v: Linear(1536→1536)                     │   │
│  │                                                         │   │
│  │  Reshape to [B, 12, seq, 128]                           │   │
│  │  Flash Attention                                        │   │
│  │  Reshape → [B, 22528, 1536]                             │   │
│  │                                                         │   │
│  │  o: Linear(1536→1536)                                  │   │
│  │  Residual: x = x + output                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FEED-FORWARD NETWORK                                    │   │
│  │                                                         │   │
│  │  x ──→ LayerNorm(1536) ──→ modulate(scale_ff, shift_ff) │   │
│  │         │                                               │   │
│  │         ├──→ ffn.0: Linear(1536→8960) ── gate branch   │   │
│  │         │    × GELU                                     │   │
│  │         └──→ ffn.2: Linear(8960→1536) × gate_ff        │   │
│  │                                                         │   │
│  │  Residual: x = x + output                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                        │
│       ▼                                                        │
│  x [B, 22528, 1536]                                            │
└────────────────────────────────────────────────────────────────┘
```

**LoRA targets**: `q, k, v, o, ffn.0, ffn.2` in each of the 30 blocks.

---

## VAE (Encoder + Decoder)

```
ENCODER                                    DECODER
────────                                   ───────
[B,3,81,512,512]                          [B,16,21,64,64]
       │                                         │
       ▼                                         ▼
Conv3d(3→128, k=3)                        Conv3d(16→512, k=3)
[B,128,81,512,512]                        [B,512,21,64,64]
       │                                         │
       ▼                                         ▼
Level 1: 2×ResBlock(128→128)             Mid: ResBlock(512)→Attn→ResBlock(512)
  Spatial ↓2                              [B,512,21,64,64]
[B,128,81,256,256]                               │
       │                                         ▼
       ▼                                  Level 4: 2×ResBlock(512→512)
Level 2: 2×ResBlock(128→256)               Spatial ↑2
  Spatial ↓2, Temporal ↓2                 [B,512,21,128,128]
[B,256,41,128,128]                               │
       │                                         ▼
       ▼                                  Level 3: 2×ResBlock(512→256)
Level 3: 2×ResBlock(256→512)               Spatial ↑2, Temporal ↑2
  Spatial ↓2, Temporal ↓2                 [B,256,41,256,256]
[B,512,21,64,64]                                 │
       │                                         ▼
       ▼                                  Level 2: 2×ResBlock(256→128)
Level 4: 2×ResBlock(512→512)               Spatial ↑2, Temporal ↑2
  (no further downsampling)               [B,128,81,512,512]
[B,512,21,64,64]                                 │
       │                                         ▼
       ▼                                  Level 1: 2×ResBlock(128→128)
Mid: ResBlock(512)→Attn→ResBlock(512)     [B,128,81,512,512]
[B,512,21,64,64]                                 │
       │                                         ▼
       ▼                                  Conv3d(128→3, k=3)
Conv3d(512→32, k=3)                       [B,3,81,512,512]
[B,32,21,64,64]                             Output Video
       │
       ▼
Split → μ,σ → Reparameterize
[B,16,21,64,64]
  Latent Space
```

---

## Text Encoder (UMT5-XXL, 24 layers)

```
"Adult Woman signing, black turtleneck, green background"
       │
       ▼
┌──────────────────────────────────┐
│ Tokenizer → [B, L≤512]          │
│       │                          │
│       ▼                          │
│ Embedding(256384, 4096)          │
│ → [B, L, 4096]                   │
│       │                          │
│       ▼                          │
│  ×24 T5 Encoder Blocks:          │
│  ┌────────────────────────────┐  │
│  │ RMSNorm(4096)              │  │
│  │ T5Attention:               │  │
│  │   q,k,v: Lin(4096→4096)   │  │
│  │   64 heads × 64 head_dim  │  │
│  │   + relative pos bias      │  │
│  │   o: Lin(4096→4096)       │  │
│  │ + Residual                 │  │
│  │                            │  │
│  │ RMSNorm(4096)              │  │
│  │ T5FFN (gated):             │  │
│  │   fc1: Lin(4096→10240)    │  │
│  │   gate: Lin(4096→10240)   │  │
│  │   × GELU                  │  │
│  │   fc2: Lin(10240→4096)    │  │
│  │ + Residual                 │  │
│  └────────────────────────────┘  │
│       │                          │
│       ▼                          │
│ Final RMSNorm + Dropout          │
│ → [B, L, 4096]                   │
└──────────────────────────────────┘
```

---

## CLIP Image Encoder (ViT-Huge-14, 32 layers)

```
Reference Image [B, 3, 224, 224]
       │
       ▼
┌──────────────────────────────────┐
│ Patch Embed: Conv2d(3→1280,      │
│              k=14, s=14)         │
│ → [B, 256, 1280]  (16×16 grid)  │
│       │                          │
│ + CLS token → [B, 257, 1280]    │
│ + Position Embed                 │
│       │                          │
│       ▼                          │
│  ×32 ViT Blocks:                 │
│  ┌────────────────────────────┐  │
│  │ LayerNorm(1280)            │  │
│  │ Self-Attention:            │  │
│  │   16 heads × 80 head_dim  │  │
│  │ + Residual                 │  │
│  │                            │  │
│  │ LayerNorm(1280)            │  │
│  │ MLP:                       │  │
│  │   Lin(1280→5120)→GELU     │  │
│  │   →Lin(5120→1280)         │  │
│  │ + Residual                 │  │
│  └────────────────────────────┘  │
│       │                          │
│       ▼                          │
│ → [B, 257, 1280]                 │
└──────────────────────────────────┘
```

---

## What Gets Trained vs Frozen

```
FROZEN (loaded once, not updated):        TRAINED (LoRA, rank=32):
─────────────────────────────────         ──────────────────────────
✗ T5 Text Encoder (~1.6B params)         ✓ DiT q,k,v,o in 30 blocks
✗ CLIP Image Encoder (~650M)                each: 1536→32→1536 (LoRA A+B)
✗ VAE Encoder + Decoder (~150M)          ✓ DiT ffn.0, ffn.2 in 30 blocks
                                            ffn.0: 1536→32→8960
                                            ffn.2: 8960→32→1536

                                         Total LoRA params per block:
                                           4×(1536×32 + 32×1536) = 393K
                                           + 1536×32+32×8960 = 336K
                                           + 8960×32+32×1536 = 336K
                                           ≈ 1.07M per block

                                         30 blocks × ~1.07M ≈ 32M trainable
```

---

## Cached Training (2-step) Data Flow

```
STEP 1 (preprocess_cache.py):              STEP 2 (train_lora_cached.py):
─────────────────────────────              ──────────────────────────────

Video ──→ VAE Encode ──→ input_latents.pt    Load cached .pt files
                          [16,21,64,64]           │
                                                  ▼
Control ─→ VAE Encode ──→ control_latents.pt  Concatenate inputs
                          [16,21,64,64]       [48,21,64,64]
                                                  │
Ref Img ─→ VAE Encode ──→ reference_latents.pt   ▼
                          [16,1,64,64]        DiT forward + backward
                                              (only DiT loaded on GPU)
Prompt ──→ T5 Encode ───→ context.pt              │
                          [L,4096]                ▼
                                              Save LoRA weights
Ref Img ─→ CLIP Encode ─→ clip_feature.pt    (.safetensors)
                          [257,1280]

Benefit: VAE + T5 + CLIP never loaded in step 2 → saves ~8GB VRAM
```

## Cached data size
```
Inspecting file: reference_latents.pt
Type: <class 'torch.Tensor'>
Shape: torch.Size([1, 16, 1, 64, 64])
Dtype: torch.float16

Inspecting file: clip_feature.pt
Type: <class 'torch.Tensor'>
Shape: torch.Size([1, 257, 1280])
Dtype: torch.float16

Inspecting file: control_latents.pt
Type: <class 'torch.Tensor'>
Shape: torch.Size([1, 16, 21, 64, 64])
Dtype: torch.float16

Inspecting file: input_latents.pt
Type: <class 'torch.Tensor'>
Shape: torch.Size([1, 16, 21, 64, 64])
Dtype: torch.float16

Inspecting file: context.pt
Type: <class 'torch.Tensor'>
Shape: torch.Size([1, 512, 4096])
Dtype: torch.float16
```
