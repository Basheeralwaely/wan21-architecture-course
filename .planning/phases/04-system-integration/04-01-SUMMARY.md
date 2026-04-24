---
phase: 04-system-integration
plan: "01"
subsystem: course
tags: [jupyter, wan2.1, diffusion, pipeline, dit, vae, t5, clip, flow-matching]

# Dependency graph
requires:
  - phase: 03-vae-track
    provides: NB-09 through NB-11 VAE encoder/decoder notebooks
  - phase: 02-dit-assembly
    provides: NB-06 through NB-08 DiT block and WanModel notebooks
  - phase: 01-dit-foundations
    provides: NB-01 through NB-05 attention and embedding primitives
provides:
  - "Course/NB-12-pipeline-system-integration.ipynb -- capstone pipeline notebook, first half"
  - "Full pipeline ASCII diagram mapping all components to prior notebooks (D-01, D-02)"
  - "Four component demos: T5, CLIP, VAE encode/decode, DiT (all with shape assertions)"
  - "SYS-01 coverage: how DiT + VAE + T5 + CLIP compose"
  - "SYS-02 coverage: tensor data flow from raw inputs to output video"
affects:
  - 04-02 (second half of NB-12: denoising loop, parameter count, summary, exercises)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Multi-module importlib setup: load wan_video_dit FIRST (image_encoder depends on it), then vae, text_encoder, image_encoder, flow_match"
    - "Real tqdm should NOT be stubbed in notebooks that also import transformers (tqdm.auto dependency)"
    - "diffsynth.diffusion namespace stub required alongside diffsynth.models stub"

key-files:
  created:
    - Course/NB-12-pipeline-system-integration.ipynb
  modified: []

key-decisions:
  - "Do not stub tqdm when transformers is also imported -- tqdm.auto required by transformers fails if tqdm is replaced"
  - "Load wan_video_dit.py first in module dependency order (wan_video_image_encoder imports flash_attention from it)"
  - "Use in_dim=16 for DiT demo cell (base T2V config) -- simpler than 48-ch Fun-Control for introductory demo"

patterns-established:
  - "Pattern: Five-module setup cell order (dit -> vae -> text_encoder -> image_encoder -> flow_match)"
  - "Pattern: ASCII diagram with [<- NB-XX] back-references on each component box"

requirements-completed: [SYS-01, SYS-02]

# Metrics
duration: 30min
completed: 2026-04-24
---

# Phase 4 Plan 01: NB-12 Pipeline System Integration (First Half) Summary

**Capstone notebook NB-12 created with full pipeline ASCII diagram, five-module setup cell, and four component demos (T5 output [1,10,4096], CLIP output [1,257,1280], VAE encode/decode [1,16,2,4,4], DiT output [1,16,4,8,8]) all executing on CPU in under 5 seconds with all assertions passing**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-04-24T20:00:00Z
- **Completed:** 2026-04-24T20:27:13Z
- **Tasks:** 1 of 1
- **Files modified:** 1

## Accomplishments
- Created `Course/NB-12-pipeline-system-integration.ipynb` as a valid Jupyter notebook (nbformat 4) with 11 cells
- Cell 0: Title, three learning objectives (SYS-01/02/03), prerequisites listing NB-01 through NB-11, concept map table mapping each notebook to its pipeline stage
- Cell 1: Setup cell loading all five modules via importlib in correct dependency order; diffsynth/diffsynth.models/diffsynth.diffusion/diffsynth.models.wan_video_camera_controller stubs registered
- Cell 3: Full pipeline ASCII diagram from raw inputs through T5/CLIP/VAE encoding, 48-channel concatenation, DiT denoising, VAE decoding to output video, with `[<- NB-XX]` back-references on every component box
- Cells 5-9: Four component demo cells (T5, CLIP, VAE, DiT) each with reduced configs, forward passes, and shape assertions
- Cell 10: Orchestration prose explaining how each component's output feeds into the next stage
- `jupyter nbconvert --execute` exits with code 0; no cell errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create NB-12 notebook with setup, pipeline diagram, and component demos** - `8ce8594` (feat)

**Plan metadata:** (committed separately after SUMMARY.md creation)

## Files Created/Modified
- `Course/NB-12-pipeline-system-integration.ipynb` - Capstone system integration notebook, first half (setup + diagram + 4 component demos)

## Decisions Made
- **Do not stub tqdm**: The `transformers` library imports `tqdm.auto` internally; if `sys.modules['tqdm']` is replaced with a simple stub module, `tqdm.auto` fails with "No module named 'tqdm.auto'; 'tqdm' is not a package". Real tqdm is available in this environment so no stub is needed. (Rule 1 auto-fix applied)
- **Load wan_video_dit FIRST in dependency order**: `wan_video_image_encoder.py` imports `flash_attention` from `.wan_video_dit` at the top level. If `wan_video_dit` is not in `sys.modules` before `wan_video_image_encoder` is loaded, the import chain fails. Order: dit -> vae -> text_encoder -> image_encoder -> flow_match.
- **Add `diffsynth.diffusion` namespace stub**: `flow_match.py` is loaded into `diffsynth.diffusion.flow_match`. The `diffsynth.diffusion` namespace module must be pre-registered in `sys.modules` before the loader runs, matching the pattern for `diffsynth` and `diffsynth.models`.
- **Use `in_dim=16` for DiT demo**: The plan specifies `in_dim=16` for the base T2V demo cell (confirmed by plan Cell 9 code). This is cleaner than the 48-channel Fun-Control config for explaining the DiT forward pass to new readers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed tqdm stub that broke transformers import chain**
- **Found during:** Task 1 (initial notebook execution test)
- **Issue:** Plan specified a tqdm stub following NB-11 pattern. However, `wan_video_text_encoder.py` imports from `transformers`, which internally imports `tqdm.auto`. When `sys.modules['tqdm']` was replaced with a simple `types.ModuleType`, `import tqdm.auto` raised `ModuleNotFoundError: No module named 'tqdm.auto'; 'tqdm' is not a package`.
- **Fix:** Removed the tqdm stub. Real tqdm is installed in this environment and works correctly. The VAE encode/decode demo runs without needing to suppress tqdm's progress bars.
- **Files modified:** Course/NB-12-pipeline-system-integration.ipynb (setup cell)
- **Verification:** `jupyter nbconvert --execute` exits with code 0; all 5 code cells produce expected output
- **Committed in:** 8ce8594 (Task 1 commit)

**2. [Rule 2 - Missing] Added diffsynth.diffusion namespace stub**
- **Found during:** Task 1 (initial setup cell design)
- **Issue:** `flow_match.py` is registered under `diffsynth.diffusion.flow_match`. Without a `diffsynth.diffusion` namespace module in `sys.modules`, Python cannot properly register the submodule.
- **Fix:** Added `_diffusion_stub = types.ModuleType('diffsynth.diffusion')` and `sys.modules['diffsynth.diffusion'] = _diffusion_stub` alongside the existing `diffsynth` and `diffsynth.models` stubs.
- **Files modified:** Course/NB-12-pipeline-system-integration.ipynb (setup cell)
- **Verification:** `FlowMatchScheduler` imports successfully; `from diffsynth.diffusion.flow_match import FlowMatchScheduler` works.
- **Committed in:** 8ce8594 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 Bug, 1 Missing Critical)
**Impact on plan:** Both fixes necessary for the setup cell to execute without errors. No scope creep.

## Issues Encountered
- NumPy 2.x compatibility warning appears in cell output (pre-existing environment issue with numexpr/sklearn/pandas). This is non-fatal -- all model imports succeed and computations produce correct results. The warning is from the environment, not from NB-12 code.

## Known Stubs
None -- all four component demos run real instantiated models with live computed parameters.

## Threat Flags
None -- NB-12 is a read-only educational notebook; no network endpoints, no file writes to untrusted paths, no user input handling.

## Next Phase Readiness
- NB-12 first half complete: setup, pipeline diagram, and four component demos (Cells 0-10)
- Plan 04-02 will add the second half: 48-channel composition demo, FlowMatchScheduler denoising loop, CFG explanation, parameter count breakdown, summary, and exercises
- No blockers for 04-02

---
*Phase: 04-system-integration*
*Completed: 2026-04-24*

## Self-Check: PASSED
