---
phase: 02-dit-assembly
plan: "02"
subsystem: course-notebooks
tags: [jupyter, wanmodel, end-to-end, gradient-checkpointing, 48-channel-concat, forward-pass]
dependency_graph:
  requires: [Course/NB-06-dit-block.ipynb, Course/NB-07-patchify-unpatchify.ipynb]
  provides: [Course/NB-08-wanmodel-forward.ipynb]
  affects: []
tech_stack:
  added: []
  patterns: [importlib-path-search-10-levels, wanmodel-reduced-block-demo, gradient-checkpointing-demo]
key_files:
  created:
    - Course/NB-08-wanmodel-forward.ipynb
  modified: []
decisions:
  - Used num_layers=3 with real production dim=1536 for CPU demo (D-07); production 30-layer config annotated in comments
  - Used has_image_input=False to avoid clip_feature dummy tensor requirement (RESEARCH.md Pitfall 3)
  - Extended path search to 10-level upward walk (matching NB-06/NB-07 deviation from 02-01-SUMMARY.md)
metrics:
  duration: ~15 minutes
  completed_date: "2026-04-24"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
---

# Phase 2 Plan 02: WanModel End-to-End Forward Pass Notebook Summary

**One-liner:** Full WanModel capstone notebook with 48-channel noise/control/ref concat, 8-step shape trace through 3-layer demo, and gradient checkpointing train-vs-eval side-by-side.

## What Was Built

### Task 1 -- NB-08: WanModel Forward Pass Notebook (commit: fd10d81)

`Course/NB-08-wanmodel-forward.ipynb` -- 16 cells (8 markdown, 8 code) covering the full WanModel end-to-end composition:

- **Cell structure:** Title + Learning Objectives + Prerequisites + Concept Map + WanModel data flow diagram -> importlib setup -> 48-channel concat demo -> WanModel init (3-layer demo) -> production config annotation -> forward pass introduction + shape trace table -> forward pass code with 8-step shape trace comments -> shape trace summary table -> gradient checkpointing introduction + condition explanation -> grad checkpoint training vs inference demo -> branching logic inspection cell -> Why grad checkpointing matters for LoRA -> summary with source references table -> exercises

- **Key demonstrations:**
  - 48-channel concat: `noise_latent (16ch)` + `control_latent (16ch)` + `ref_latent (16ch)` created separately, then `torch.cat([...], dim=1)` with assertion (D-08)
  - WanModel instantiation: `dim=1536, in_dim=48, ffn_dim=8960, out_dim=16, num_layers=3` (165M params demo; production 30-layer config shown as commented annotation)
  - Forward pass: `model(x_48, timestep, context)` in `model.eval()` + `torch.no_grad()`; `assert out.shape == torch.Size([B, 16, F, H, W])` passes
  - 8-step shape trace as inline comments (time_embedding -> time_projection -> text_embedding -> patchify -> freqs assembly -> 3x DiTBlock -> Head -> unpatchify)
  - Gradient checkpointing: `model.train()` -> `model(x_train, ..., use_gradient_checkpointing=True)` then `model.eval()` -> `model(x_48, ...)` in `torch.no_grad()`; both paths produce same shape
  - `model.training` state inspection showing propagation to submodules (blocks[0].training)
  - Explicit pitfall documented: eval mode + use_gradient_checkpointing=True silently runs direct path

- **All assertions pass on CPU; execution verified with no cell errors**

## Standards Compliance

| Standard | NB-08 | Notes |
|----------|-------|-------|
| STD-01: Prerequisites cell | PASS | NB-06, NB-07 listed in Learning Objectives and Concept Map |
| STD-02: Inline shape annotations | PASS | Every tensor operation annotated with shape comments |
| STD-03: CPU execution under 5s | PASS | 3-layer model at F=4,H=8,W=8 runs in ~0.042s (verified in RESEARCH.md) |
| STD-04: assert statements | PASS | x_48 shape assert, out.shape assert, out_gc/out_eval shape assert |
| STD-05: Markdown before code cells | PASS | Every code cell preceded by a markdown explanation cell |
| STD-06: Source line citations | PASS | Lines 274-328 (WanModel init), 359-416 (forward), 381-385 (freqs), 393-412 (checkpoint) |
| STD-07: Real diffsynth imports | PASS | `from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d` |

## Deviations from Plan

### Auto-applied Context (Not Bugs)

**1. [Context Match] 10-level path search applied from 02-01-SUMMARY.md deviation**
- **Found during:** Setup cell implementation
- **Issue:** The 02-01-SUMMARY.md documented that NB-05's 2-level upward path search fails from git worktrees and was extended to 10-level walk in NB-06/NB-07.
- **Action:** Applied the same 10-level upward walk in NB-08 setup cell (consistent with sibling notebooks).
- **Files modified:** `Course/NB-08-wanmodel-forward.ipynb` (setup cell)

**2. [Context Match] WanModel forward pass branching differs from plan comments**
- **Found during:** Reading actual wan_video_dit.py lines 393-412
- **Issue:** The plan's gradient checkpointing description referenced `create_custom_forward` wrapping, which is the actual implementation. The plan also notes `x.requires_grad_(True)` is set before the loop. Both documented accurately in the notebook's inline comments.
- **Action:** Notebook comments cite the actual implementation accurately including the `x = x.requires_grad_(True)` step at line 394.

**3. [Context Match] `sinusoidal_embedding_1d` signature is `(freq_dim, timestep)` not `(timestep, freq_dim)`**
- **Found during:** Reading wan_video_dit.py line 370
- **Issue:** The plan's shape trace comment showed `sinusoidal([B, freq_dim=256])` -- the actual call is `sinusoidal_embedding_1d(self.freq_dim, timestep)`. The notebook's Exercise 2 correctly calls `sinusoidal_embedding_1d(256, timestep)`.
- **Action:** Exercise 2 uses the correct argument order.

## Known Stubs

None -- all demonstrations use real `WanModel` imported from `diffsynth.models.wan_video_dit`. Dummy tensors are used for inputs (as required by STD-03 portability constraint), but all model classes are real.

## Threat Flags

None -- notebook is a read-only educational file with no network calls, no user data, and no secrets.

## Self-Check

Files exist:
- `Course/NB-08-wanmodel-forward.ipynb`: FOUND (in worktree)

Commits exist:
- `fd10d81`: FOUND (feat(02-02): create NB-08 WanModel end-to-end forward pass notebook)

Execution verified:
- NB-08: `jupyter nbconvert --execute` completed successfully with zero errors
- All 8 code cells ran to completion
- All assertions passed: `x_48.shape`, `out.shape`, `out_gc.shape == out_eval.shape`
- Output shapes confirmed: `[1, 48, 4, 8, 8]` in, `[1, 16, 4, 8, 8]` out
- Gradient checkpointing paths both confirmed via shape assertion

Static checks (15/15 PASS): cell_count, has_WanModel, has_48ch_concat, has_num_layers_3, has_production_30, has_forward_pass, has_shape_assert, has_grad_checkpoint, has_train_eval, has_shape_trace, has_exercises, has_summary, has_source_citation, has_prerequisites, has_importlib

## Self-Check: PASSED
