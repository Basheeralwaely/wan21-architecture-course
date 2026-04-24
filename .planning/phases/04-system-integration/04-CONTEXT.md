# Phase 4: System Integration - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Create 1 Jupyter notebook (NB-12) in `Course/` that converges the DiT track (NB-01–NB-08) and VAE track (NB-09–NB-11) into the full WanVideoPipeline. Readers finish this notebook able to trace any tensor from raw input (video + text + reference image) through encoding, denoising, and decoding to output video — and understand how all four model components (DiT, VAE, T5, CLIP) compose.

</domain>

<decisions>
## Implementation Decisions

### Data Flow Diagram
- **D-01:** Single large ASCII diagram showing the full pipeline end-to-end — from raw inputs (video, text prompt, reference image) through T5/CLIP encoding, VAE latent encoding, 48-channel concatenation, DiT denoising, VAE decoding, to output video. Tensor shapes annotated at each stage.
- **D-02:** Each component box in the diagram includes notebook back-references (e.g., "VAE Encoder (NB-10)", "DiT (NB-06–NB-08)") so readers know where to go for deeper understanding. The diagram acts as a roadmap linking the capstone to all prior notebooks.

### Denoising Loop
- **D-03:** Walkthrough with scheduler — show the FlowMatchScheduler timestep setup, walk through one denoising step conceptually (noise prediction → scheduler step → updated latents), and note that production uses 50 steps. Explain flow matching velocity prediction at a level accessible to someone encountering it for the first time. Cover classifier-free guidance (positive vs negative predictions). Do NOT go into full flow matching theory or ELBO math (out of scope per REQUIREMENTS.md), but explain enough that a reader can follow the denoising code.

### Runnable Demos
- **D-04:** Component-by-component demos — create minimal dummy instances of each component (T5 text encoder, CLIP image encoder, VAE, DiT) separately, run each with dummy input, show its output shape. Then show the orchestration (how outputs connect) conceptually via the ASCII diagram. Keeps each cell fast and focused under STD-03's 5-second limit.
- **D-05:** Live parameter count computation — instantiate each component and compute `sum(p.numel() for p in model.parameters())`. Produce a breakdown table showing each component's contribution (DiT, VAE, T5, CLIP) with percentages. Consistent with NB-06's LoRA parameter analysis approach.

### Claude's Discretion
- Import/setup strategy (same pattern as Phase 1/2/3 — Claude established it)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within the 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- Whether to use reduced dimensions for text encoder and CLIP (like Phase 2 used reduced block count for DiT)
- How much of the WanVideoPipeline unit architecture to expose — the 20+ unit classes (PromptEmbedder, ImageEmbedderCLIP, FunControl, etc.) are implementation detail; Claude decides how much to surface
- Notebook narrative arc — top-down (pipeline overview first, zoom into components) vs bottom-up (recap, then compose)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model source code
- `diffsynth/pipelines/wan_video.py` — WanVideoPipeline class (lines 32–82: component composition and unit architecture), `__call__` method (lines 178–334: full inference flow with denoising loop), `model_fn_wan_video` (model forward function used in denoising)
- `diffsynth/models/wan_video_dit.py` — WanModel (lines 273–416: DiT with 30 blocks, 48-channel input)
- `diffsynth/models/wan_video_vae.py` — WanVideoVAE: Encoder3d (lines 517–617), Decoder3d (lines 736–838)
- `diffsynth/models/wan_video_text_encoder.py` — WanTextEncoder, HuggingfaceTokenizer (T5/UMT5 text embedding)
- `diffsynth/models/wan_video_image_encoder.py` — WanImageEncoder (CLIP vision encoder)
- `diffsynth/diffusion/flow_match.py` — FlowMatchScheduler (timestep scheduling for denoising loop)

### Existing notebooks (Phase 1+2+3 output — follow same template)
- `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` — Template-setting notebook (standards reference)
- `Course/NB-06-dit-block.ipynb` — DiT Block + LoRA param analysis (reference for parameter counting approach)
- `Course/NB-07-patchify-unpatchify.ipynb` — DiT patchify (back-reference for diagram)
- `Course/NB-08-wanmodel-forward.ipynb` — Full WanModel + 48-ch concat (back-reference for DiT section)
- `Course/NB-10-vae-encoder.ipynb` — VAE Encoder (back-reference for encoding stage)
- `Course/NB-11-vae-decoder-patchify.ipynb` — VAE Decoder (back-reference for decoding stage)

### Architecture documentation
- `model_architecture.md` — Full pipeline diagram, parameter counts, tensor shapes
- `.planning/codebase/ARCHITECTURE.md` — Layer-by-layer architecture analysis with data flow

### Requirements
- `.planning/REQUIREMENTS.md` — SYS-01 through SYS-03 (Phase 4 content requirements), STD-01 through STD-07 (notebook standards)

### Prior phase context
- `.planning/phases/01-dit-foundations/01-CONTEXT.md` — Phase 1 decisions (template structure D-01 through D-04)
- `.planning/phases/02-dit-assembly/02-CONTEXT.md` — Phase 2 decisions (ASCII diagrams, reduced blocks, 48-ch concat)
- `.planning/phases/03-vae-track/03-CONTEXT.md` — Phase 3 decisions (VAE primitives, encoder/decoder walkthroughs)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `WanVideoPipeline` (wan_video.py:32–82): Orchestrates all components — holds references to dit, vae, text_encoder, image_encoder with unit-based processing architecture
- `FlowMatchScheduler` (flow_match.py): Provides timestep scheduling for denoising loop — `set_timesteps()`, `step()` methods
- `WanTextEncoder` (wan_video_text_encoder.py): T5/UMT5 text encoder producing context embeddings
- `WanImageEncoder` (wan_video_image_encoder.py): CLIP ViT producing 257-token image features
- Phase 1–3 notebooks in `Course/`: 11 notebooks establishing template pattern and covering all sub-components
- `model_architecture.md`: Pre-computed architecture analysis with parameter counts and tensor shapes

### Established Patterns
- einops `rearrange` for tensor reshaping — consistent across all prior notebooks
- Reduced model dimensions for CPU execution (Phase 2: 2–3 blocks instead of 30)
- Component-by-component demo style (each notebook instantiates the subject with minimal config)
- ASCII diagrams for architecture visualization (Phase 2 D-02/D-05, Phase 3 D-01)
- Live parameter counting with formatted table output (NB-06)

### Integration Points
- NB-12 back-references ALL prior notebooks — the capstone that ties NB-01–NB-11 together
- Pipeline `__call__` shows the full inference orchestration (units → denoising loop → decode)
- 48-channel concat already demonstrated in NB-08 — NB-12 contextualizes it within the full pipeline

</code_context>

<specifics>
## Specific Ideas

- Denoising section should be written for someone encountering flow matching for the first time — the user specifically noted they don't fully understand it, so this is a key teaching opportunity
- The diagram should serve as a "course map" showing which notebooks taught which components

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-system-integration*
*Context gathered: 2026-04-24*
