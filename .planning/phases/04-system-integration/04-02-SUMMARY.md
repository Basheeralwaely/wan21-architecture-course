---
phase: 04-system-integration
plan: "02"
subsystem: course
tags: [jupyter, wan2.1, flow-matching, denoising-loop, cfg, parameter-count, capstone]

# Dependency graph
requires:
  - phase: 04-system-integration
    plan: "01"
    provides: NB-12 first half (setup, pipeline diagram, four component demos, Cells 0-10)
provides:
  - "Course/NB-12-pipeline-system-integration.ipynb -- complete capstone notebook (22 cells)"
  - "Section 3: 48-channel composition demo with correct x/y separation (SYS-02)"
  - "Section 4: FlowMatchScheduler denoising loop walkthrough with CFG explanation (D-03)"
  - "Section 5: Live demo + verified production parameter count table (D-05, SYS-03)"
  - "Summary with Key Takeaways, Source References, and Course Map (NB-01 through NB-11)"
  - "Exercises section with 3 modification exercises"
affects:
  - "(none -- NB-12 is the final notebook in the course)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FlowMatchScheduler('Wan') usage: set_timesteps(n, denoising_strength=1.0, shift=5.0) then scheduler.step(pred, t, latents)"
    - "Flow matching step formula: prev_sample = sample + velocity * (sigma_next - sigma_current); negative delta moves toward clean image"
    - "CFG combination: noise_pred = nega + cfg_scale * (posi - nega)"

key-files:
  created: []
  modified:
    - Course/NB-12-pipeline-system-integration.ipynb

key-decisions:
  - "Remove '780M' string entirely rather than noting it as incorrect -- acceptance criteria required no '780M' anywhere; replaced with prose about understatement and measured value"
  - "Use single DiT pass (no CFG) in denoising demo loop for clarity; explain CFG in separate prose cell (Cell 16)"
  - "All 11 new cells appended in a single python script and committed together (Tasks 1+2 in one commit)"

requirements-completed: [SYS-02, SYS-03]

# Metrics
duration: 35min
completed: 2026-04-24
---

# Phase 4 Plan 02: NB-12 Pipeline System Integration (Second Half) Summary

**NB-12 completed as a 22-cell capstone notebook: 48-channel composition demo showing correct x=[B,16,...] + y=[B,32,...] WanModel.forward signature, FlowMatchScheduler 3-step denoising loop running with sigma progression printed, CFG formula explained, and verified production parameter table (DiT 1.56B, VAE 127M, T5 5.68B, CLIP 632M, total 8.00B asserted) -- all cells execute on CPU with exit code 0**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-24T20:00:00Z
- **Completed:** 2026-04-24T20:35:11Z
- **Tasks:** 2 of 2
- **Files modified:** 1

## Accomplishments

- Appended 11 cells (11-21) to the existing NB-12 notebook from Plan 01 (cells 0-10)
- **Cell 11 (markdown):** Section 3 -- 48-Channel Input Composition. Explains that WanModel.forward takes `x=[B,16,...]` and `y=[B,32,...]` separately; back-references NB-08; explains the internal `cat([x, y], dim=1)` that produces the 48-channel input to `patch_embedding Conv3d(48->1536)`
- **Cell 12 (code):** Demo of `y = cat(control_latents, ref_embedding)` assembly with `assert y.shape == (B, 32, T, H, W)` passing
- **Cell 13 (markdown):** Section 4 -- Denoising Loop. Written for a reader encountering flow matching for the first time (per CONTEXT.md specifics). Covers: what is flow matching vs DDPM noise prediction, the sigma schedule from 1.0 to ~0.093, the step formula with explanation of why negative delta moves toward clean image, and CFG (positive + negative prompt combination)
- **Cell 14 (code):** FlowMatchScheduler 50-step demo showing first/last 5 timesteps and sigmas
- **Cell 15 (code):** 3-step denoising loop with formatted table output (step, timestep, sigma, sigma_next, latent_norm); `assert latents.shape` passes
- **Cell 16 (markdown):** CFG prose with `noise_pred_posi`/`noise_pred_nega` combination formula; explains cfg_scale=1.0, 7.5, 20.0 implications
- **Cell 17 (markdown):** Section 5 intro -- parameter count breakdown referencing NB-06 approach
- **Cell 18 (code):** Live parameter count for demo-config models (`sum(p.numel() for p in model.parameters())` loop)
- **Cell 19 (code):** Production parameter table with verified values; `assert total == 8_004_481_844` passes
- **Cell 20 (markdown):** Summary with Key Takeaways (5 bullets), Source References table (10 entries), Course Map table linking NB-01 through NB-12
- **Cell 21 (markdown):** 3 exercises (DiT block count, sigma schedule, CFG scale analysis)
- `jupyter nbconvert --execute` exits with code 0; all assertions pass

## Task Commits

1. **Tasks 1 + 2: Add all second-half cells (11-21) to NB-12** - `0d58676` (feat)

Note: Both tasks were implemented in a single python script run and committed together. The commit message documents Task 1's cells (11-16); Task 2's cells (17-21) are included in the same commit. This is a minor process deviation from the plan's expectation of one commit per task.

**Plan metadata:** (committed separately after SUMMARY.md creation)

## Files Created/Modified

- `Course/NB-12-pipeline-system-integration.ipynb` - Capstone system integration notebook, now complete (22 cells, second half: 48-ch demo, denoising loop, CFG, parameter count, summary, exercises)

## Decisions Made

- **Remove "780M" string entirely**: The acceptance criteria required no "780M" anywhere in the notebook. Rather than document the model_architecture.md discrepancy by quoting the wrong figure, replaced the mention with neutral prose ("understates the DiT parameter count") and referenced RESEARCH.md for verification. The measured values (1.56B Fun-Control) are what appear in Cell 19.
- **Single-pass denoising demo**: Cell 15 runs the DiT once per step (no CFG) for pedagogical clarity. CFG is explained in prose (Cell 16) with the source code pattern from wan_video.py lines 301-309. This avoids running the DiT 6 times (2 × 3 steps) for demo purposes.
- **All new cells committed together**: Tasks 1 and 2 were implemented and committed in a single operation. The plan expected separate per-task commits; the deviation is harmless since both tasks are fully complete and verified.

## Deviations from Plan

### Auto-fixed Issues

None -- all acceptance criteria were met on first execution after fixing the "780M" string in cells 17 and 19.

### Process Deviations

**1. Tasks 1 and 2 committed in a single commit**
- **Found during:** Implementation (all 11 cells written in one python script)
- **Issue:** The plan expected one commit per task; cells 11-16 (Task 1) and cells 17-21 (Task 2) were written together and committed as `0d58676`
- **Impact:** None -- all acceptance criteria for both tasks pass; no correctness issue
- **Commit:** 0d58676

**2. Initial "780M" reference required removal**
- **Found during:** Task 2 acceptance criteria check
- **Issue:** Cells 17 and 19 contained "~780M" string (noting the model_architecture.md error) which triggered the acceptance criteria "does NOT contain '780M'"
- **Fix:** Replaced with "understates the DiT parameter count" in Cell 17 and "understates the param count" comment in Cell 19
- **Files modified:** Course/NB-12-pipeline-system-integration.ipynb (committed in 0d58676)

---

**Total deviations:** 2 minor process deviations (no functional/correctness impact)
**Impact on plan:** Zero -- all success criteria and acceptance criteria met

## Issues Encountered

- The pre-existing numpy 1.x/2.x compatibility warning generates an `AttributeError: _ARRAY_API not found` in the Cell 1 output during `nbconvert --execute`. This was already documented in Plan 01 SUMMARY as a non-fatal environment issue. All model imports succeed; the "Setup complete" message prints correctly despite the error annotation.

## Known Stubs

None -- all parameter counts are live-computed from real model instances. The production parameter table uses verified values from RESEARCH.md, not estimates.

## Threat Flags

None -- NB-12 is a read-only educational notebook; no network endpoints, no file writes to untrusted paths, no user input handling.

## Self-Check

---

## Self-Check: PASSED

- `Course/NB-12-pipeline-system-integration.ipynb` exists and has 22 cells
- Commit `0d58676` exists in git log
- `jupyter nbconvert --execute` exits with code 0
- `assert total == 8_004_481_844` passes (Cell 19 output confirmed)
- `assert latents.shape` passes (Cell 15 output confirmed)
- "780M" does not appear in source cells
- Course Map back-references NB-01 through NB-11

*Phase: 04-system-integration*
*Completed: 2026-04-24*
