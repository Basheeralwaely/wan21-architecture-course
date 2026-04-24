# Phase 3: VAE Track - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Create 3 Jupyter notebooks (NB-09 through NB-11) in `Course/` that teach the Wan2.1 video VAE architecture: CausalConv3d primitives, ResBlock, and AttentionBlock (NB-09), Encoder3d downsampling pathway with reparameterization (NB-10), and Decoder3d upsampling with VAE vs DiT patchify disambiguation (NB-11). Readers finish this phase able to trace video through the full encode/decode path and distinguish the two patchify semantics.

</domain>

<decisions>
## Implementation Decisions

### CausalConv3d Teaching (NB-09)
- **D-01:** Visual-first explanation of asymmetric temporal padding — ASCII diagram showing how left-only padding prevents future frames from leaking into current frame computation, then derive the padding formula `(2*kernel_t_pad, 0)` from the diagram. Consistent with Phase 2's ASCII diagram approach.
- **D-02:** Full conv pipeline walkthrough for ResBlock — walk the entire residual sequential layer by layer: RMSNorm → SiLU → CausalConv3d → RMSNorm → SiLU → Dropout → CausalConv3d, showing how each layer transforms the tensor. Include the skip connection path (identity when in_dim==out_dim, learned 1×1 CausalConv3d when dimensions change).
- **D-03:** Include AttentionBlock in NB-09 alongside CausalConv3d and ResBlock — keeps all VAE primitives in one notebook so NB-10/NB-11 can reference back without interrupting the encoder/decoder narrative.

### Encoder Walkthrough (NB-10)
- **D-04:** Full level-by-level trace of the 4-level downsampling path with a shape table showing spatial, temporal, and channel dimensions at each stage (Input → Level 0 → Level 1 → Level 2 → Level 3 → Middle → Head → Latent).
- **D-05:** Show Resample module internals — walk through the spatial downsample path (ZeroPad2d + strided Conv2d) and temporal downsample path (CausalConv3d with stride). These are new operations not fully covered in NB-09.
- **D-06:** Code-focused reparameterization coverage — show the mu/log_var split from conv1 output, the `reparameterize()` function (`eps * std + mu`), and the scale normalization. Briefly note WHY it enables backprop through sampling, without diving into VAE ELBO loss theory.

### Decoder + Patchify Disambiguation (NB-11)
- **D-07:** Mirror-plus-differences approach for decoder — start with "the decoder mirrors the encoder" framing, show the upsampling shape table (reverse of NB-10's table), then focus on what's DIFFERENT: Resample upsample mode, extra ResBlock per level (`num_res_blocks + 1`), reversed `dim_mult`, channel halving in upsample Resample.
- **D-08:** Dedicated side-by-side comparison section for VAE patchify vs DiT patchify — comparison table and ASCII diagrams showing: VAE patchify = einops channel rearrangement (deterministic, no learned parameters, spatial → channels), DiT patchify = Conv3d with stride (learned projection, video patches → token sequence). Back-reference NB-07.

### Claude's Discretion
- Import/setup strategy (same pattern as Phase 1/2 — Claude established it)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within the 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- feat_cache handling in code examples — VAE source is heavily interleaved with feat_cache streaming (out of scope per REQUIREMENTS.md). Claude decides how to handle in notebook code: strip it, briefly acknowledge, or use simplified wrappers.
- Whether to reduce encoder/decoder depth for runnable cells (e.g., fewer levels or reduced dims) to meet the 5-second STD-03 limit

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model source code
- `diffsynth/models/wan_video_vae.py` — Full VAE implementation: CausalConv3d (lines 33-52), RMS_norm (lines 55-70), Resample (lines 82-196), patchify/unpatchify (lines 199-224), ResidualBlock (lines 267-301), AttentionBlock (lines 304-342), Encoder3d (lines 517-617), Decoder3d (lines 736-838), VideoVAE_ (lines 951-1055)

### Existing notebooks (Phase 1+2 output — follow same template)
- `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` — Template-setting notebook (standards reference)
- `Course/NB-07-patchify-unpatchify.ipynb` — DiT patchify/unpatchify (back-reference for NB-11 disambiguation)

### Architecture documentation
- `model_architecture.md` — Full pipeline diagram, VAE structure with tensor shapes
- `.planning/codebase/ARCHITECTURE.md` — Layer-by-layer architecture analysis with data flow

### Requirements
- `.planning/REQUIREMENTS.md` — VAE-01 through VAE-05 (Phase 3 content requirements), STD-01 through STD-07 (notebook standards)

### Prior phase context
- `.planning/phases/01-dit-foundations/01-CONTEXT.md` — Phase 1 decisions (template structure D-01 through D-04, Claude's discretion areas)
- `.planning/phases/02-dit-assembly/02-CONTEXT.md` — Phase 2 decisions (ASCII diagrams, reduced block count for execution)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CausalConv3d` (wan_video_vae.py:33-52): Core primitive — asymmetric temporal padding with feat_cache support for streaming
- `ResidualBlock` (wan_video_vae.py:267-301): Skip connection with optional 1×1 channel projection, two CausalConv3d layers with RMSNorm and SiLU
- `AttentionBlock` (wan_video_vae.py:304-342): Causal self-attention with single head — used in encoder/decoder middle blocks
- `Resample` (wan_video_vae.py:82-196): Spatial (strided Conv2d) and temporal (CausalConv3d stride + PixelUnshuffle) downsampling/upsampling
- `Encoder3d` (wan_video_vae.py:517-617): 4-level downsampling encoder with dim_mult=[1,2,4,4], z_dim=16
- `Decoder3d` (wan_video_vae.py:736-838): Mirror of encoder with upsampling, extra ResBlock per level
- `patchify`/`unpatchify` (wan_video_vae.py:199-224): Einops-based channel rearrangement (NOT learned projection)
- Phase 1+2 notebooks in `Course/`: 8 notebooks establishing the template pattern

### Established Patterns
- einops `rearrange` for tensor reshaping — used extensively in VAE patchify/unpatchify
- CausalConv3d wraps nn.Conv3d with modified padding for causal inference
- RMS_norm (VAE version) differs from WanRMSNorm (DiT version) — channel_first=True, different broadcasting
- feat_cache streaming throughout VAE code — out of scope but pervasive in source

### Integration Points
- NB-09 back-references NB-01 (RMSNorm concept, though VAE uses its own RMS_norm variant)
- NB-10 back-references NB-09 (CausalConv3d, ResBlock, AttentionBlock as building blocks)
- NB-11 back-references NB-10 (encoder as mirror) and NB-07 (DiT patchify for disambiguation)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-vae-track*
*Context gathered: 2026-04-24*
