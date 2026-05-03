# WAN 2.2 Architecture Course

A hands-on deep dive into the WAN 2.2 video diffusion model. This course covers the complete architecture -- from primitive operations through the full pipeline -- using reduced-config notebooks that run on CPU without model weights or GPU.

## Who This Is For

ML engineers and researchers who want to understand how WAN 2.2 works internally. You should be comfortable with:

- **PyTorch basics:** `nn.Module`, tensors, forward passes, autograd
- **Transformer attention concepts:** queries, keys, values, multi-head attention
- **Basic diffusion model awareness:** noise schedules, denoising process (no deep expertise needed)

What you **don't** need:
- GPU hardware
- Model weights
- Training or fine-tuning experience

## How It Works

Every notebook instantiates real model components with reduced configs so cells run on CPU in under 5 seconds without downloading model weights.

- All notebooks use `dim=384, num_heads=4, head_dim=96` (instead of production dimensions like `dim=5120`)
- Every notebook runs on CPU in under 5 seconds per cell
- Shape assertions at every transformation verify correct data flow
- Source code from `diffsynth/models/` is loaded directly via `importlib` (no package install needed)
- The reduced config satisfies S2V divisibility constraints: `head_dim//2 = 48`, `48 % 3 = 0`

## Course Map

The course has six tracks that build on each other. See the suggested reading order at the end of this section.

### Track 1: DiT Primitives (NB-01 to NB-05)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-01** | RMSNorm, Sinusoidal Embeddings, Modulate | Normalization, timestep encoding, conditioning primitive |
| **NB-02** | QKV Projections and Head Layout | Multi-head attention inputs, dual-backend layouts |
| **NB-03** | 3D Rotary Position Embeddings | Frequency bands, grid assembly, S2V comparison |
| **NB-04** | Self-Attention and Cross-Attention | RoPE-enhanced attention, CLIP/T5 dual-stream |
| **NB-05** | adaLN-Zero Modulation | Timestep conditioning, gated residuals, six parameters |

### Track 2: DiT Assembly (NB-06 to NB-08)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-06** | DiTBlock | All primitives wired together into a single transformer block |
| **NB-07** | Patchify and Unpatchify | Video frames to patch sequences and back via Conv3d |
| **NB-08** | WanModel Forward Pass | Complete diffusion backbone end-to-end |

### Track 3: VAE (NB-09 to NB-11)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-09** | CausalConv3d, ResBlock, Attention | Temporal causal padding, residual blocks, spatial attention |
| **NB-10** | VAE Encoder | 4-level spatial downsampling with reparameterization |
| **NB-11** | VAE Decoder | Latent reconstruction to full-resolution video |

### Track 4: VACE Control (NB-12 to NB-13)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-12** | VaceWanAttentionBlock | Control encoder block, stack/unstack accumulation |
| **NB-13** | VACE Integration | Hint pre-computation, additive injection into DiT |

### Track 5: S2V Pipeline (NB-14 to NB-17)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-14** | Audio Encoding | CausalConv1d, weighted layer aggregation |
| **NB-15** | FramePackMotioner | Multi-scale patchify, per-token RoPE computation |
| **NB-16** | S2V DiTBlock and AudioInjector | Dual timestep modulation, audio cross-attention |
| **NB-17** | WanS2VModel Forward Pass | Complete sketch-to-video architecture |

### Track 6: Integration (NB-18)

| Notebook | Topic | What You Learn |
|----------|-------|----------------|
| **NB-18** | Pipeline Orchestration | Model dispatch, scheduler loop, CFG, VAE decode |

### Suggested Reading Order

- **Tracks 1-2** are sequential -- each notebook builds on the previous ones. Start with NB-01 and work through NB-08 in order.
- **Track 3** (VAE) can be read after Track 1. It is independent of Track 2 content.
- **Tracks 4 and 5** both require Track 2 (understanding of DiTBlock and WanModel).
- **Track 6** requires Tracks 3, 4, and 5 (all components must be understood before seeing how they connect).

## Prerequisites

- Python 3.8+
- PyTorch 2.x
- einops
- JupyterLab or Jupyter Notebook
- The `diffsynth/` source code (included in this repository under `diffsynth/`)

No GPU or model weights required.

## Getting Started

1. Clone the repository:
   ```
   git clone <repo-url>
   cd wan22-architecture-course
   ```

2. Install dependencies:
   ```
   pip install torch einops jupyterlab
   ```

3. Launch Jupyter:
   ```
   cd Course
   jupyter lab
   ```

4. Open NB-01 and start with Track 1.

Each notebook lists its specific prerequisites and builds on concepts from earlier notebooks.

## Architecture Overview

```
Text Prompt ──> T5 Text Encoder ──> text_emb [B, S_text, 4096]
                                          │
Reference Image ──> CLIP ViT-H/14 ──> clip_emb [B, 257, 1280]
                                          │
                                    ┌─────┴─────┐
                                    │ Projection │ (dim mapping)
                                    └─────┬─────┘
                                          │
                                    context [B, 257+S_text, dim]
                                          │
Video Frames ──> VAE Encoder ──> latents [B, C, F, H, W]
                                    │
                              ┌─────┴─────┐
                              │  Patchify  │ (Conv3d)
                              └─────┬─────┘
                                    │
                              patches [B, S, dim]
                                    │
                         ┌──────────┴──────────┐
                         │    DiT Backbone      │
                         │  ┌────────────────┐  │
                         │  │  DiTBlock x N   │  │
                         │  │  ┌──────────┐  │  │
                         │  │  │ adaLN-Zero│  │  │
                         │  │  │ Self-Attn │  │  │
                         │  │  │ Cross-Attn│  │  │
                         │  │  │ FFN       │  │  │
                         │  │  └──────────┘  │  │
                         │  └────────────────┘  │
                         │                      │
                         │  VACE path:          │
                         │  hints ──> VaceEnc   │
                         │  ──> additive inject │
                         │                      │
                         │  S2V path:           │
                         │  audio+pose ──>      │
                         │  FramePack+AudioInj  │
                         └──────────┬──────────┘
                                    │
                              ┌─────┴─────┐
                              │ Unpatchify │
                              └─────┬─────┘
                                    │
                              ┌─────┴─────┐
                              │VAE Decoder │
                              └─────┬─────┘
                                    │
                              Video Frames
```

## Key Numbers

| Component | Parameters (Production) | Role |
|-----------|------------------------|------|
| T5 Text Encoder | 5.68B | Text conditioning |
| CLIP ViT-H/14 | 632M | Image conditioning |
| DiT Backbone (base) | ~1.3B | Diffusion denoising |
| VACE Control Encoder | ~200M | Control signal encoding |
| S2V Model | ~1.3B | Sketch-to-video |
| VAE | ~200M | Encode/decode video frames |

Parameter counts are approximate for the production model. This course uses reduced configs (`dim=384`) for CPU execution.
