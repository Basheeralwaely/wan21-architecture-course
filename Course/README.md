# Wan2.1 Architecture Course

A 12-notebook deep dive into the internals of the Wan2.1-Fun-V1.1-1.3B-Control video diffusion model. Each notebook walks through the actual [DiffSynth](https://github.com/modelscope/DiffSynth-Studio) source code, building understanding bottom-up — from individual attention primitives to the complete 8-billion-parameter pipeline.

## Who This Is For

Intermediate PyTorch practitioners who want to understand how a production video diffusion model works internally. You should be comfortable with tensors, `nn.Module`, and forward passes. No prior knowledge of diffusion models or DiT architectures is required — those concepts are explained as they appear.

## How It Works

Every notebook instantiates real model components with reduced configs (fewer layers, smaller dimensions) so cells run on CPU in under 5 seconds without downloading model weights. Shape assertions verify data flow at every stage so you can trust the demonstrations match the real architecture.

## Course Map

The course has four tracks that build on each other:

### Track 1: DiT Primitives (NB-01 to NB-05)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-01** | RMSNorm, Sinusoidal Embeddings, Modulate | Why Wan2.1 uses RMSNorm over LayerNorm, how timesteps become frequency vectors, and the scale/shift mechanics behind adaptive normalization |
| **NB-02** | QKV Projections and Head Layout | How attention blocks organize queries, keys, and values across heads, with dual backend support (flash_attn vs PyTorch SDPA) |
| **NB-03** | 3D Rotary Position Embeddings (RoPE) | How video transformers encode spatial-temporal position using unequal frequency bands for frame, height, and width |
| **NB-04** | Self-Attention and Cross-Attention | The dual attention mechanism — self-attention with RoPE on Q/K, cross-attention splitting CLIP image tokens from T5 text tokens |
| **NB-05** | adaLN-Zero Modulation | How each DiT block is conditioned on the diffusion timestep via six learned parameters (shift, scale, gate for both attention and FFN) |

### Track 2: DiT Assembly (NB-06 to NB-08)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-06** | The DiT Block | How all five primitives wire together into a single transformer block, parameter distribution, and LoRA target identification |
| **NB-07** | Patchify, Unpatchify, and Head | How video frames become token sequences via Conv3d and how denoised tokens reconstruct video through einops rearrangement |
| **NB-08** | WanModel End-to-End Forward Pass | The complete 8-step forward pass from 48-channel latent input through 30 DiT blocks, plus gradient checkpointing and memory tradeoffs |

### Track 3: VAE (NB-09 to NB-11)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-09** | CausalConv3d, ResidualBlock, AttentionBlock | The VAE building blocks — causal temporal masking, skip connections, and per-frame spatial self-attention |
| **NB-10** | VAE Encoder3d | How raw video is compressed through 4-level downsampling (spatial /8) to produce latent tensors, with reparameterization for training |
| **NB-11** | VAE Decoder3d and Patchify Disambiguation | How denoised latents reconstruct full-resolution video, and why VAE patchify (einops) differs from DiT patchify (Conv3d) |

### Track 4: System Integration (NB-12)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-12** | Full Pipeline Integration | How DiT (1.56B), VAE (127M), T5 (5.68B), and CLIP (632M) compose into the 8.0B-parameter pipeline — including the 48-channel input composition, flow matching denoising loop, and classifier-free guidance |

## Prerequisites

- Python 3.8+
- PyTorch (CPU is sufficient)
- Jupyter Notebook or JupyterLab
- The `diffsynth/` source code (included in this repository)

No GPU or model weights required.

## Getting Started

```bash
pip install -r requirements.txt
cd Course
jupyter notebook
```

Start with **NB-01** and work through in order. Each notebook lists its specific prerequisites and builds on concepts from earlier notebooks.

## Architecture Overview

```
Text Prompt ──► T5 Text Encoder (5.68B) ──► context [B, seq, 4096]
                                                    │
Reference Image ──► CLIP Encoder (632M) ──► clip [B, 257, 1280]
                                                    │
Control Video ──► VAE Encode (127M) ──► control_latents [B, 16, F, H, W]
                                                    │
Noise ──────────────────────────────► noise [B, 16, F, H, W]
                                                    │
                              ┌─────────────────────┘
                              ▼
                    48-ch Concatenation
                    (noise + control + ref)
                              │
                              ▼
                    DiT (1.56B, 30 blocks)
                    ── Patchify ──► Self-Attn ──► Cross-Attn ──► FFN ──► Unpatchify
                              │
                              ▼
                    Flow Match Denoising (50 steps)
                              │
                              ▼
                    VAE Decode (127M)
                              │
                              ▼
                        Output Video
```

## Key Numbers

| Component | Parameters | Role |
|-----------|-----------|------|
| T5 Text Encoder | 5.68B | Text prompt → context embeddings |
| DiT (WanModel) | 1.56B | Denoising transformer (30 blocks) |
| CLIP Image Encoder | 632M | Reference image → visual features |
| VAE | 127M | Video ↔ latent space compression |
| **Total** | **8.0B** | |
