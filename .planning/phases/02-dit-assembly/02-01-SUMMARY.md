---
phase: 02-dit-assembly
plan: "01"
subsystem: course-notebooks
tags: [jupyter, dit-block, patchify, unpatchify, head, lora]
dependency_graph:
  requires: [Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb, Course/NB-04-self-cross-attention.ipynb, Course/NB-05-adaln-zero-modulation.ipynb]
  provides: [Course/NB-06-dit-block.ipynb, Course/NB-07-patchify-unpatchify.ipynb]
  affects: [Course/NB-08-wanmodel-forward.ipynb]
tech_stack:
  added: []
  patterns: [importlib-path-search-up-10-levels, einops-rearrange, conv3d-patchify]
key_files:
  created:
    - Course/NB-06-dit-block.ipynb
    - Course/NB-07-patchify-unpatchify.ipynb
  modified: []
decisions:
  - Extended setup cell path search from 2-level to 10-level upward walk to handle git worktrees
metrics:
  duration: ~20 minutes
  completed_date: "2026-04-24"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
---

# Phase 2 Plan 01: DiT Block and Patchify/Unpatchify Notebooks Summary

**One-liner:** DiTBlock composition walkthrough (6 sub-modules, 3D freqs assembly, 90.7% LoRA coverage) and Conv3d patchify/unpatchify/Head round-trip with t vs t_mod distinction.

## What Was Built

### Task 1 -- NB-06: DiT Block Notebook (commit: 5ee1380)

`Course/NB-06-dit-block.ipynb` -- 15 cells covering the complete DiT Block composition:

- **Cell structure:** Markdown title with ASCII block diagram -> importlib setup -> sub-module inventory -> 3D freqs assembly -> forward pass -> annotated forward pass -> LoRA parameter counts -> 30-block scaling -> summary -> exercises
- **Key demonstrations:**
  - DiTBlock instantiation with `has_image_input=True, dim=1536, num_heads=12, ffn_dim=8960`
  - Full 3-band freqs assembly via `precompute_freqs_cis_3d(head_dim)` producing shape `[64, 1, 64]`
  - Annotated 4-stage forward pass with back-references to NB-01, NB-04, NB-05
  - Per-sub-module LoRA parameter counts: 90.7% coverage (46,422,272 / 51,163,904 per block)
  - 30-block full-model scaling: 1,392,668,160 total LoRA parameters
- **All assertions pass on CPU; execution verified under 30s timeout**

### Task 2 -- NB-07: Patchify/Unpatchify/Head Notebook (commit: cf3e52e)

`Course/NB-07-patchify-unpatchify.ipynb` -- 13 cells covering three concepts:

- **Cell structure:** Markdown title with ASCII spatial-to-sequence diagram -> importlib setup -> patchify (Conv3d + rearrange) -> unpatchify (einops rearrange) -> shape round-trip verification -> Head module -> Head pipeline explanation -> summary -> exercises
- **Key demonstrations:**
  - `nn.Conv3d(48, 1536, kernel_size=(1,2,2), stride=(1,2,2))` producing `[1, 1536, 4, 4, 4]` from `[1, 48, 4, 8, 8]`
  - Exact unpatchify rearrange string `'b (f h w) (x y z c) -> b c (f x) (h y) (w z)'` from `WanModel.unpatchify` line 353
  - Shape round-trip assertion: input spatial `(4, 8, 8)` == output spatial `(4, 8, 8)`
  - Head module with explicit `t` vs `t_mod` distinction (2-parameter adaLN, not 6-parameter)
  - Head sub-module counts: norm (0 params, no affine), head linear (98,368), modulation (3,072)
- **All assertions pass on CPU; execution verified under 30s timeout**

## Standards Compliance

| Standard | NB-06 | NB-07 |
|----------|-------|-------|
| STD-01: Prerequisites cell | PASS (NB-01, NB-04, NB-05) | PASS (NB-01, NB-06) |
| STD-02: Inline shape annotations | PASS | PASS |
| STD-03: CPU execution under 5s | PASS (~0.020s DiTBlock forward) | PASS (<0.001s Head forward) |
| STD-04: assert statements | PASS (freqs, out, block shapes) | PASS (x_seq, x_video, head out) |
| STD-05: Markdown before code cells | PASS | PASS |
| STD-06: Source line citations | PASS (lines 197-231, 381-385) | PASS (lines 340-357, 254-270) |
| STD-07: Real diffsynth imports | PASS (DiTBlock, GateModule, etc.) | PASS (WanModel, Head) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended setup cell path search from 2-level to 10-level upward walk**
- **Found during:** Task 1 implementation
- **Issue:** The plan specified "copy VERBATIM from NB-05 cell 1 (the 6-level upward path search importlib pattern)" but the actual NB-05 setup cell only searches 2 levels (`[_here, _here.parent, _here.parent.parent]`). From the git worktree's `Course/` directory, `diffsynth/` is 4 levels up -- this would fail with `FileNotFoundError`.
- **Fix:** Changed the path search to a loop that walks up to 10 levels, stopping at filesystem root. This correctly finds the project root from both the worktree (4 levels up) and the main repo (1 level up).
- **Files modified:** Both `NB-06-dit-block.ipynb` and `NB-07-patchify-unpatchify.ipynb` setup cells
- **Verification:** Confirmed execution succeeds from `Course/` in main repo (standard location). The loop approach handles all depth variations without fragility.

## Known Stubs

None -- all demonstrations use real DiTBlock, WanModel, and Head instances imported from `diffsynth.models.wan_video_dit`. No placeholder data flows to displayed output.

## Threat Flags

None -- notebooks are read-only educational files with no network calls, no user data, and no secrets.

## Self-Check

Files exist:
- `Course/NB-06-dit-block.ipynb`: FOUND
- `Course/NB-07-patchify-unpatchify.ipynb`: FOUND

Commits exist:
- `5ee1380`: FOUND (feat(02-01): create NB-06 DiT Block composition notebook)
- `cf3e52e`: FOUND (feat(02-01): create NB-07 patchify/unpatchify/Head notebook)

Execution verified:
- NB-06: jupyter nbconvert --execute completed successfully, all 7 code cells ran, all assertions passed
- NB-07: jupyter nbconvert --execute completed successfully, all 5 code cells ran, all assertions passed

## Self-Check: PASSED
