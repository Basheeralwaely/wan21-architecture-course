# Requirements: Wan2.1 Model Architecture Course

**Defined:** 2026-04-24
**Core Value:** Clear, annotated code walkthroughs that make the Wan2.1 architecture understandable — each notebook builds on the last so readers can trace how basic components compose into the full model.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Notebook Standards

- [ ] **STD-01**: Every notebook has a prerequisite statement at the top listing prior notebooks and concepts assumed
- [ ] **STD-02**: Every reshape/projection operation has inline tensor shape annotations (e.g., `# [B, S, N, D] -> [B, N, S, D]`)
- [ ] **STD-03**: Every notebook has runnable dummy tensor cells that execute on CPU in under 5 seconds
- [ ] **STD-04**: Every notebook has `assert output.shape == expected` verification cells after key operations
- [ ] **STD-05**: Prose-before-code — every code cell is preceded by a markdown explanation of the concept
- [ ] **STD-06**: Source file and line number references point to the actual `diffsynth/models/` implementation
- [ ] **STD-07**: Real `diffsynth` classes are imported and run (not toy reimplementations)

### DiT Track — Primitives

- [ ] **DIT-01**: NB-01 covers RMSNorm — why RMSNorm over LayerNorm, implementation walkthrough, shape verification
- [ ] **DIT-02**: NB-01 covers sinusoidal timestep embeddings — frequency computation, shape output
- [ ] **DIT-03**: NB-01 covers the `modulate` function — scale/shift/gate mechanics
- [ ] **DIT-04**: NB-02 covers QKV projections — linear projections, multi-head splitting via einops
- [ ] **DIT-05**: NB-02 covers multi-head attention layout — `(B,S,N,D)` vs `(B,N,S,D)` conventions shown side by side
- [ ] **DIT-06**: NB-03 covers 3D RoPE — the three-axis head-dimension split (temporal/height/width frequency bands)
- [ ] **DIT-07**: NB-03 covers `precompute_freqs_cis_3d` — frequency computation, complex representation, dtype requirements (float64)

### DiT Track — Modules

- [ ] **DIT-08**: NB-04 covers SelfAttention — full forward pass with RoPE integration, flash attention fallback
- [ ] **DIT-09**: NB-04 covers CrossAttention — text context conditioning, `has_image_input` dual-stream path (257 CLIP tokens)
- [ ] **DIT-10**: NB-05 covers adaLN-Zero — six modulation parameters, gate=0 identity behavior demonstration, zero-init rationale
- [ ] **DIT-11**: NB-05 covers the per-block `modulation` parameter — learned additive offset on time embedding

### DiT Track — Assembly

- [ ] **DIT-12**: NB-06 covers DiT Block — composition of self-attention + cross-attention + FFN + adaLN conditioning
- [ ] **DIT-13**: NB-06 includes parameter count breakdown identifying LoRA target modules (q, k, v, o, ffn.0, ffn.2)
- [ ] **DIT-14**: NB-07 covers patchify — Conv3d learned projection with patch size (1,2,2), video-to-token conversion
- [ ] **DIT-15**: NB-07 covers unpatchify — token-to-video reconstruction, shape recovery
- [ ] **DIT-16**: NB-08 covers full WanModel end-to-end — 30 blocks, 48-channel input composition (noise + control + reference)
- [ ] **DIT-17**: NB-08 covers gradient checkpointing stripping for inference context

### VAE Track

- [ ] **VAE-01**: NB-09 covers CausalConv3d — asymmetric temporal padding derivation, why causal convolution for video
- [ ] **VAE-02**: NB-09 covers ResBlock — skip connections with optional channel projection
- [ ] **VAE-03**: NB-10 covers VAE Encoder — downsampling pathway, latent space dimensions, reparameterization
- [ ] **VAE-04**: NB-11 covers VAE Decoder — upsampling pathway, latent-to-video reconstruction
- [ ] **VAE-05**: NB-11 disambiguates VAE `patchify` from DiT `patchify` — different semantics explicitly called out

### System Integration

- [ ] **SYS-01**: NB-12 covers full pipeline overview — how DiT + VAE + text encoder + CLIP compose into WanVideoPipeline
- [ ] **SYS-02**: NB-12 shows data flow diagram — from raw inputs through encoding, denoising, decoding to output video
- [ ] **SYS-03**: NB-12 includes parameter count summary for entire model (~780M DiT params + VAE + encoders)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Content

- **ADV-01**: Standalone 3D RoPE deep-dive notebook with 1D-to-3D progression
- **ADV-02**: FlashAttention benchmarking notebook comparing attention backends
- **ADV-03**: Interactive ipywidgets sliders for exploring attention patterns
- **ADV-04**: Training loop walkthrough notebook (how LoRA training uses these components)
- **ADV-05**: Inference pipeline walkthrough (50-step denoising loop)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Modifying existing codebase files | Explicit constraint — read-only codebase |
| Loading real model weights | Portability — notebooks must run without GPU or model downloads |
| Training tutorials | Scope is architecture only, not training workflow |
| Inference walkthroughs | Covered by existing scripts, not architectural teaching |
| Beginner PyTorch fundamentals | Target audience is intermediate — knows tensors, modules, forward passes |
| VAE feat_cache streaming mechanism | Inference optimization, not architecture — explicitly out of scope |
| Flow matching diffusion theory | Mathematical framework — would require its own course |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| STD-01 | TBD | Pending |
| STD-02 | TBD | Pending |
| STD-03 | TBD | Pending |
| STD-04 | TBD | Pending |
| STD-05 | TBD | Pending |
| STD-06 | TBD | Pending |
| STD-07 | TBD | Pending |
| DIT-01 | TBD | Pending |
| DIT-02 | TBD | Pending |
| DIT-03 | TBD | Pending |
| DIT-04 | TBD | Pending |
| DIT-05 | TBD | Pending |
| DIT-06 | TBD | Pending |
| DIT-07 | TBD | Pending |
| DIT-08 | TBD | Pending |
| DIT-09 | TBD | Pending |
| DIT-10 | TBD | Pending |
| DIT-11 | TBD | Pending |
| DIT-12 | TBD | Pending |
| DIT-13 | TBD | Pending |
| DIT-14 | TBD | Pending |
| DIT-15 | TBD | Pending |
| DIT-16 | TBD | Pending |
| DIT-17 | TBD | Pending |
| VAE-01 | TBD | Pending |
| VAE-02 | TBD | Pending |
| VAE-03 | TBD | Pending |
| VAE-04 | TBD | Pending |
| VAE-05 | TBD | Pending |
| SYS-01 | TBD | Pending |
| SYS-02 | TBD | Pending |
| SYS-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 0
- Unmapped: 32

---
*Requirements defined: 2026-04-24*
*Last updated: 2026-04-24 after initial definition*
