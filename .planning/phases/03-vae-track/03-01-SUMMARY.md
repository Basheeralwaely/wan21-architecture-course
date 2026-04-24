---
phase: 03-vae-track
plan: "01"
subsystem: Course/notebooks
tags: [vae, causalconv3d, residualblock, attentionblock, rms-norm, jupyter]
dependency_graph:
  requires: []
  provides: [NB-09-causalconv3d-resblock-attn]
  affects: [NB-10, NB-11]
tech_stack:
  added: []
  patterns: [importlib-path-search, tqdm-stub, reduced-dim-demo, cpu-execution]
key_files:
  created:
    - Course/NB-09-causalconv3d-resblock-attn.ipynb
  modified: []
decisions:
  - "Fixed plan inaccuracy: AttentionBlock only zeros proj.weight (not proj.bias); assertion and prose updated to match actual source (wan_video_vae.py:319)"
metrics:
  duration: "332s (5m 32s)"
  completed: "2026-04-24"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 3 Plan 01: NB-09 VAE Primitives Notebook Summary

## One-Liner

CausalConv3d asymmetric temporal padding walkthrough with ResidualBlock full pipeline and AttentionBlock per-frame spatial self-attention, all importlib-loaded from `wan_video_vae.py`.

## What Was Built

`Course/NB-09-causalconv3d-resblock-attn.ipynb` — 16-cell Jupyter notebook teaching the three VAE primitive building blocks. Follows the Phase 1/2 template (STD-01 through STD-07): title/LO/prerequisites/concept map, setup cell, prose→code interleaved sections, summary, exercises.

**Cell map:**
- Cell 0 (md): Title, learning objectives, prerequisites (NB-01), concept map linking to NB-10/NB-11
- Cell 1 (code): Setup — tqdm stub, 10-level path search for `wan_video_vae.py`, importlib load, imports
- Cell 2 (md): CausalConv3d — prose explaining asymmetric temporal padding
- Cell 3 (md): ASCII diagram showing padding formula derivation (D-01)
- Cell 4 (code): CausalConv3d demo — `_padding` print + shape assertion `(1,8,4,8,8)`
- Cell 5 (code): Padding formula verification for k=3, k=5, k=1
- Cell 6 (md): RMS_norm — VAE vs DiT comparison table
- Cell 7 (code): `RMS_norm(dim=8, images=False)` demo — gamma shapes for 5D and 4D tensors
- Cell 8 (md): ResidualBlock — full pipeline walkthrough prose (D-02)
- Cell 9 (code): ResidualBlock architecture inspection — sequential layer list
- Cell 10 (code): Both skip paths — Identity (same dim) and CausalConv3d (diff dim)
- Cell 11 (md): AttentionBlock — per-frame spatial self-attention explanation (D-03)
- Cell 12 (code): AttentionBlock forward trace with shape annotations
- Cell 13 (code): Zero-weight proj verification — checks `proj.weight == 0`, corrects plan inaccuracy about `proj.bias`
- Cell 14 (md): Summary — key takeaways + source references table
- Cell 15 (md): Exercises (3 exercises: padding formula, channel expansion, parameter count)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect AttentionBlock zero-init assertion**
- **Found during:** Task 1 — notebook execution
- **Issue:** Plan Cell 13 claimed `proj.bias all zeros: True` and asserted `residual < 1e-5` (near-identity). The actual source (`wan_video_vae.py:319`) only calls `nn.init.zeros_(self.proj.weight)` — `proj.bias` uses `nn.Conv2d` default initialization (non-zero Kaiming uniform). At initialization `proj(features) = 0 * features + bias = bias`, so output is NOT pure identity.
- **Fix:** Updated Cell 12 to show `proj.bias all zeros: False`. Rewrote Cell 13 to assert `proj.weight == 0` (the actual invariant), explain that `proj.bias` is non-zero, and verify shape preservation instead of near-zero residual. Updated Cell 11 markdown and Cell 0 learning objective to accurately describe the initialization behavior.
- **Files modified:** `Course/NB-09-causalconv3d-resblock-attn.ipynb`
- **Commit:** 8e16ca2 (same task commit — fix applied before task commit)

## Verification Results

```
PASS: cell_count (16 cells)
PASS: has_CausalConv3d
PASS: has_ResidualBlock
PASS: has_AttentionBlock
PASS: has_RMS_norm
PASS: has_padding_tuple
PASS: has_shape_assert
PASS: has_ascii_diagram
PASS: has_skip_paths
PASS: has_exercises
PASS: has_summary
PASS: has_source_citation
PASS: has_prerequisites
PASS: has_importlib
PASS: has_tqdm_stub
PASS: has_zero_init
Overall: PASS (16/16)
jupyter nbconvert --execute: SUCCESS (no errors, all assertions pass)
```

## Known Stubs

None. All code cells use real `diffsynth.models.wan_video_vae` classes imported via importlib. No hardcoded data, no mock outputs.

## Threat Flags

None. Notebook is a read-only educational file with dummy tensors and public architecture walkthrough code.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Create NB-09 | 8e16ca2 | feat(03-01): create NB-09 VAE primitives notebook |

## Self-Check: PASSED

- `Course/NB-09-causalconv3d-resblock-attn.ipynb`: FOUND
- Commit `8e16ca2`: FOUND (verified via git log)
- Notebook executes via `jupyter nbconvert --execute`: PASSED
- Structural checks 16/16: PASSED
