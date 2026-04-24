# Roadmap: Wan2.1 Model Architecture Course

## Overview

Four phases build the course bottom-up, following the architecture's natural dependency graph. Phase 1 establishes the DiT primitives and the notebook standards template that every later phase replicates. Phase 2 assembles those primitives into the full DiT model. Phase 3 builds the parallel VAE track independently. Phase 4 converges both tracks into the complete pipeline walkthrough. A reader finishing Phase 4 can trace any tensor from raw input to decoded output video.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: DiT Foundations** - NB-01 through NB-05: primitives (RMSNorm, sinusoidal embeddings, modulate, QKV, 3D RoPE, attention modules, adaLN) with notebook standards established
- [ ] **Phase 2: DiT Assembly** - NB-06 through NB-08: DiT Block composition, patch embedding, and full WanModel end-to-end
- [ ] **Phase 3: VAE Track** - NB-09 through NB-11: CausalConv3d primitives, VAE Encoder downsampling, VAE Decoder upsampling
- [ ] **Phase 4: System Integration** - NB-12: full pipeline data flow converging DiT + VAE + text encoder + CLIP

## Phase Details

### Phase 1: DiT Foundations
**Goal**: Readers can run and understand the five primitive/module notebooks (NB-01 through NB-05) that form the atomic building blocks of the DiT, with all notebook standards established and verified
**Depends on**: Nothing (first phase)
**Requirements**: STD-01, STD-02, STD-03, STD-04, STD-05, STD-06, STD-07, DIT-01, DIT-02, DIT-03, DIT-04, DIT-05, DIT-06, DIT-07, DIT-08, DIT-09, DIT-10, DIT-11
**Success Criteria** (what must be TRUE):
  1. A reader can open NB-01 cold and run every cell on CPU in under 5 seconds, observing correct shapes for RMSNorm, sinusoidal embeddings, and the modulate function
  2. A reader can open NB-02 and observe Q, K, V projection shapes side-by-side in both `(B,S,N,D)` and `(B,N,S,D)` conventions via runnable cells
  3. A reader can open NB-03 and run `precompute_freqs_cis_3d` with dummy inputs, seeing the three-axis frequency split across temporal/height/width head dimensions
  4. A reader can open NB-04 and run both SelfAttention and CrossAttention full forward passes with RoPE applied and dual-stream CLIP path exercised
  5. A reader can open NB-05 and observe gate=0 identity behavior for adaLN-Zero modulation, with the learned per-block modulation offset demonstrated
**Plans:** 3 plans
Plans:
- [x] 01-01-PLAN.md — Setup + NB-01: RMSNorm, sinusoidal embedding, modulate (template-setting notebook)
- [x] 01-02-PLAN.md — NB-02: QKV projections/head layout + NB-03: 3D RoPE
- [x] 01-03-PLAN.md — NB-04: SelfAttention/CrossAttention + NB-05: adaLN-Zero modulation

### Phase 2: DiT Assembly
**Goal**: Readers can trace a full DiT inference pass through a composed DiT Block, patch embedding/unpatchify, and the 30-block WanModel with 48-channel concatenated input
**Depends on**: Phase 1
**Requirements**: DIT-12, DIT-13, DIT-14, DIT-15, DIT-16, DIT-17
**Success Criteria** (what must be TRUE):
  1. A reader can open NB-06 and see the self-attention + cross-attention + FFN + adaLN composition in one forward pass, with a parameter count cell identifying LoRA target modules
  2. A reader can open NB-07 and observe the patchify/unpatchify shape round-trip: `(B,C,F,H,W)` → flattened sequence → back to video dimensions
  3. A reader can open NB-08 and run a full WanModel forward pass with dummy 48-channel input (noise + control + reference), seeing gradient checkpointing stripped for inference
**Plans:** 2 plans
Plans:
- [x] 02-01-PLAN.md — NB-06: DiT Block composition + LoRA analysis, NB-07: Patchify/Unpatchify/Head
- [x] 02-02-PLAN.md — NB-08: WanModel end-to-end forward pass with 48-channel input and gradient checkpointing

### Phase 3: VAE Track
**Goal**: Readers can run the three VAE notebooks (NB-09 through NB-11) covering causal convolution primitives, encoder downsampling, and decoder upsampling, and can distinguish VAE patchify from DiT patchify
**Depends on**: Phase 1 (standards template established; NB-09 can start in parallel after Phase 1 completes)
**Requirements**: VAE-01, VAE-02, VAE-03, VAE-04, VAE-05
**Success Criteria** (what must be TRUE):
  1. A reader can open NB-09 and run CausalConv3d with a dummy video tensor, observing the asymmetric temporal padding and understanding why causal structure prevents temporal leakage
  2. A reader can open NB-10 and trace the encoder's 4-level downsampling from input video dimensions to latent space, including the reparameterization split
  3. A reader can open NB-11 and trace the decoder's upsampling back to pixel frames, with VAE patchify semantics explicitly contrasted against DiT patchify
**Plans:** 2 plans
Plans:
- [x] 03-01-PLAN.md — NB-09: CausalConv3d, ResidualBlock, AttentionBlock (VAE primitives)
- [x] 03-02-PLAN.md — NB-10: VAE Encoder downsampling + NB-11: VAE Decoder upsampling and patchify disambiguation
**UI hint**: no

### Phase 4: System Integration
**Goal**: Readers can trace the complete Wan2.1 pipeline from raw inputs through encoding, denoising, and decoding — understanding how DiT and VAE compose and what the 48-channel concatenated input represents
**Depends on**: Phase 2, Phase 3
**Requirements**: SYS-01, SYS-02, SYS-03
**Success Criteria** (what must be TRUE):
  1. A reader can open NB-12 and follow the data flow diagram from raw video + text inputs through VAE encoding, DiT denoising, and VAE decoding back to output video
  2. A reader can see a parameter count summary cell covering the entire model (~780M DiT params + VAE + encoders) with each component's contribution labeled
  3. A reader can identify exactly how the 48-channel DiT input is composed (`cat(noise_latents, control_latents, ref_latents, dim=1)`) and why multi-modal conditioning uses this concatenation approach
**Plans:** 2 plans
Plans:
- [x] 04-01-PLAN.md — NB-12 setup, pipeline ASCII diagram, and component-by-component demos (T5, CLIP, VAE, DiT)
- [x] 04-02-PLAN.md — NB-12 48-channel composition, denoising loop walkthrough, parameter count table, summary, and exercises

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. DiT Foundations | 3/3 | Complete | 2026-04-24 |
| 2. DiT Assembly | 2/2 | Complete | 2026-04-24 |
| 3. VAE Track | 0/2 | Planned | - |
| 4. System Integration | 0/2 | Planned | - |
