# Phase 1: DiT Foundations - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Create 5 Jupyter notebooks (NB-01 through NB-05) in `Course/` that teach the atomic building blocks of the Wan2.1 DiT. Covers RMSNorm, sinusoidal embeddings, modulate, QKV projections, multi-head attention layout, 3D RoPE, SelfAttention, CrossAttention, and adaLN-Zero modulation. All notebook standards (STD-01 through STD-07) are established here as the template for later phases.

</domain>

<decisions>
## Implementation Decisions

### Notebook Template Structure
- **D-01:** Rich template structure — every notebook includes: learning objectives (3+ bullets), prerequisites (prior notebooks + assumed concepts), concept map (simple list showing where each concept reappears in later notebooks), prose→code→verify sections, summary (key takeaways), and exercises section
- **D-02:** Concept map uses simple list format — "RMSNorm → used in DiT blocks (NB-06)" — plain text showing forward references, not visual diagrams
- **D-03:** Exercises are modification tasks only — "change X and observe what happens" style (e.g., swap RMSNorm for LayerNorm, change head count, alter frequency bands). No comprehension questions.
- **D-04:** 2-3 exercises per notebook, each targeting a different concept covered in that notebook

### Claude's Discretion
- Import/setup strategy — how notebooks access `diffsynth` classes (sys.path, pip install -e, or PYTHONPATH assumption)
- Source code presentation style — inline annotated snippets vs. file references
- Explanation depth — how much mathematical/theoretical context to include alongside code walkthroughs
- Exact prose length and tone in markdown cells
- Verification cell design beyond shape assertions (numerical checks, visualization)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model source code
- `diffsynth/models/wan_video_dit.py` — DiT implementation: RMSNorm, sinusoidal_embedding_1d, modulate, precompute_freqs_cis_3d, rope_apply, flash_attention, SelfAttention, CrossAttention, WanAttentionBlock classes
- `diffsynth/models/general_modules.py` — Shared layer definitions used across models

### Architecture documentation
- `model_architecture.md` — Full pipeline diagram, DiT block structure with tensor shapes, parameter counts
- `.planning/codebase/ARCHITECTURE.md` — Layer-by-layer architecture analysis with data flow

### Requirements
- `.planning/REQUIREMENTS.md` — STD-01 through STD-07 (notebook standards), DIT-01 through DIT-11 (content requirements per notebook)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `diffsynth/models/wan_video_dit.py`: Contains all primitive functions and classes NB-01 through NB-05 will import and demonstrate — RMSNorm (as WanRMSNorm), sinusoidal_embedding_1d, modulate, precompute_freqs_cis/precompute_freqs_cis_3d, rope_apply, flash_attention, SelfAttention, CrossAttention classes
- `model_architecture.md`: Contains detailed ASCII diagrams and tensor shape annotations that can inform notebook prose and diagrams

### Established Patterns
- einops `rearrange` used throughout for tensor reshaping — notebooks should use the same convention
- Flash attention with multiple backend fallbacks (flash_attn_3 → flash_attn_2 → sage_attn → PyTorch SDPA) — notebooks will use compatibility_mode=True for CPU execution
- Complex-number RoPE representation via `torch.view_as_complex` / `torch.view_as_real` — requires float64 dtype

### Integration Points
- `Course/` directory does not exist yet — needs to be created
- Notebooks import from `diffsynth.models.wan_video_dit` — PYTHONPATH must include project root
- No existing notebooks or course materials to maintain consistency with

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

*Phase: 01-dit-foundations*
*Context gathered: 2026-04-24*
