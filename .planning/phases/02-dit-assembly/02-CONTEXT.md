# Phase 2: DiT Assembly - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Create 3 Jupyter notebooks (NB-06 through NB-08) in `Course/` that assemble the Phase 1 primitives into composed systems: the DiT Block (NB-06), patch embedding and unpatchify (NB-07), and the full WanModel end-to-end forward pass (NB-08). Readers finish this phase able to trace a complete DiT inference pass from 48-channel input through 30 blocks to output video latents.

</domain>

<decisions>
## Implementation Decisions

### Composition Narrative (NB-06)
- **D-01:** Bottom-up buildup approach — start with `DiTBlock.__init__` showing sub-module wiring, then walk through `forward()` line-by-line with back-references to prior notebooks ("recall from NB-05: this is the adaLN modulation"), ending with a full forward pass cell and shape verification
- **D-02:** Include an ASCII block diagram at the top of NB-06 showing the DiT Block data flow: input → adaLN → self-attn → cross-attn → FFN → output with residual connections marked

### LoRA Parameter Analysis (NB-06)
- **D-03:** Counts + rationale depth — show parameter counts per sub-module (q, k, v, o, ffn.0, ffn.2) with percentages, explain WHY these layers are LoRA targets (task-specific representations), but no LoRA math or rank decomposition details
- **D-04:** Both per-block and full-model scope — show the detailed per-block breakdown first, then scale up to the full 30-block model total

### Patchify Presentation (NB-07)
- **D-05:** ASCII diagrams showing spatial-to-sequence mapping — visual intuition of how Conv3d with stride (1,2,2) carves video into patches and flattens to a token sequence, consistent with NB-06's diagram approach
- **D-06:** Include the Head module in NB-07 alongside patchify/unpatchify — keeps the input projection / output projection story complete in one notebook

### Full Model Execution (NB-08)
- **D-07:** Reduced block count (2-3 blocks) for runnable forward pass cells to stay under 5-second STD-03 limit. Show real 30-block config in a separate annotation cell
- **D-08:** Explicit 48-channel concat demo — create separate dummy tensors for noise (16ch), control (16ch), and reference (16ch) latents, show `torch.cat` on dim=1, then feed into model. Reader sees exactly how the 48 channels compose
- **D-09:** Gradient checkpointing shown as training vs inference side-by-side — demonstrate that training mode wraps blocks in `torch.utils.checkpoint.checkpoint()` while eval mode runs blocks directly, explaining memory savings vs recomputation tradeoff

### Claude's Discretion
- Import/setup strategy (same as Phase 1 — Claude established the pattern)
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions
- Exercise design within the 2-3 modification exercises per notebook (Phase 1 template: D-03, D-04)
- Whether to reduce `dim` alongside block count in NB-08, or keep dim=1536 with only block reduction

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model source code
- `diffsynth/models/wan_video_dit.py` — DiTBlock (lines 196-231: __init__ and forward), WanModel (lines 273-416: full model with patchify/unpatchify/forward), Head (lines 254-270), MLP (lines 234-251)
- `diffsynth/models/general_modules.py` — Shared layer definitions

### Existing notebooks (Phase 1 output — follow same template)
- `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` — Template-setting notebook (standards reference)
- `Course/NB-02-qkv-projections-head-layout.ipynb` — QKV projections
- `Course/NB-03-3d-rope.ipynb` — 3D RoPE
- `Course/NB-04-self-cross-attention.ipynb` — SelfAttention + CrossAttention
- `Course/NB-05-adaln-zero-modulation.ipynb` — adaLN-Zero modulation

### Architecture documentation
- `model_architecture.md` — Full pipeline diagram, DiT block structure with tensor shapes, parameter counts
- `.planning/codebase/ARCHITECTURE.md` — Layer-by-layer architecture analysis with data flow

### Requirements
- `.planning/REQUIREMENTS.md` — DIT-12 through DIT-17 (Phase 2 content requirements), STD-01 through STD-07 (notebook standards)

### Prior phase context
- `.planning/phases/01-dit-foundations/01-CONTEXT.md` — Phase 1 decisions (template structure D-01 through D-04, Claude's discretion areas)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DiTBlock` class (wan_video_dit.py:196-231): Composes SelfAttention + CrossAttention + FFN + adaLN modulation — the core subject of NB-06
- `WanModel` class (wan_video_dit.py:273-416): Full model with patchify (Conv3d), unpatchify (einops rearrange), 30 DiT blocks, gradient checkpointing — subjects of NB-07 and NB-08
- `Head` class (wan_video_dit.py:254-270): Final projection with adaLN modulation — pairs with patchify/unpatchify in NB-07
- Phase 1 notebooks in `Course/`: 5 notebooks establishing the template pattern (learning objectives, prerequisites, concept map, prose→code→verify, summary, exercises)

### Established Patterns
- einops `rearrange` for tensor reshaping — used in patchify/unpatchify
- Compatibility mode for attention (flash_attn fallback to PyTorch SDPA) — relevant for CPU execution in notebooks
- Complex-number RoPE via `torch.view_as_complex` / `torch.view_as_real` — NB-06 block forward needs freqs input
- Phase 1 notebook template: rich structure with learning objectives, prerequisites, concept map, prose→code→verify sections, summary, 2-3 modification exercises

### Integration Points
- NB-06 back-references NB-01 (modulate), NB-04 (SelfAttention, CrossAttention), NB-05 (adaLN-Zero)
- NB-07 back-references NB-06 (DiT Block is what processes the patchified tokens)
- NB-08 back-references all prior notebooks — full model composes everything

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

*Phase: 02-dit-assembly*
*Context gathered: 2026-04-24*
