---
phase: 01-dit-foundations
plan: 01
subsystem: notebooks
tags: [jupyter, pytorch, rmsnorm, sinusoidal-embedding, modulate, diffsynth, importlib, nbformat]

# Dependency graph
requires: []
provides:
  - "Course/ directory with NB-01-rmsnorm-sinusoidal-modulate.ipynb"
  - "Template-setting notebook covering RMSNorm (line 101), sinusoidal_embedding_1d (line 68), modulate (line 64)"
  - "All 7 notebook standards (STD-01 through STD-07) demonstrated and established"
  - "importlib path-search pattern for finding diffsynth/ across checkout scenarios"
affects: [01-02, 01-03, later-phases]

# Tech tracking
tech-stack:
  added: [jupyter, nbformat]
  patterns:
    - "importlib.util.spec_from_file_location to load wan_video_dit.py without diffsynth/__init__.py"
    - "Upward directory search for PROJECT_ROOT to handle worktree/normal checkout scenarios"
    - "Cell ordering: Title+LO+Prerequisites+ConceptMap → Setup → Markdown+Code pairs → Summary+Exercises"

key-files:
  created:
    - "Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb"
  modified: []

key-decisions:
  - "Use pathlib upward search (not fixed '..') to locate diffsynth/ — handles git worktrees and non-standard layouts"
  - "importlib direct-load pattern established as the shared import setup for all 5 notebooks in the phase"
  - "Notebook has 15 cells following D-01 template: markdown+learning objectives, setup, 3 concept sections (each with markdown intro + code demo), comparison cells, summary, exercises"

patterns-established:
  - "STD-07: importlib.util.spec_from_file_location with wan_video_camera_controller stub — copy verbatim to NB-02..NB-05"
  - "STD-04: assert output.shape == torch.Size([...]) + print after every tensor operation"
  - "STD-06: # Source: diffsynth/models/wan_video_dit.py, line N on first use of each symbol"
  - "STD-02: inline shape comment [B, S, dim] on every operation with non-obvious output shape"
  - "D-03/D-04: exercises as modification tasks — change X observe what happens, 3 per notebook targeting different concepts"

requirements-completed: [STD-01, STD-02, STD-03, STD-04, STD-05, STD-06, STD-07, DIT-01, DIT-02, DIT-03]

# Metrics
duration: 4min
completed: 2026-04-24
---

# Phase 1 Plan 01: DiT Foundations NB-01 Summary

**Gold-standard template notebook covering RMSNorm (float32 upcast, weight-only), sinusoidal_embedding_1d (float64 precision, dtype preservation), and modulate (adaLN identity property) — 15 cells, all 7 STD standards, executes CPU-only in 3.5 seconds**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-24T11:30:56Z
- **Completed:** 2026-04-24T11:34:53Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Created `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` with 15 cells covering all three DiT primitives (DIT-01, DIT-02, DIT-03)
- Established the importlib direct-load pattern (with upward directory search for PROJECT_ROOT) as the canonical setup cell for all 5 Phase 1 notebooks
- All 7 notebook standards (STD-01 through STD-07) demonstrated: prerequisites header, inline shape annotations, CPU-safe dummy tensors, shape assertions, prose-before-code ordering, source citations, real diffsynth class imports via importlib
- Notebook executes end-to-end on CPU in 3.5 seconds with all assertions passing (RMSNorm identity not tested, modulate identity confirmed numerically, sinusoidal cos/sin structure verified)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Course directory and install jupyter** - (no separate commit — empty directory; jupyter already installed)
2. **Task 2: Write NB-01 notebook** - `9acc7cd` (feat)

## Files Created/Modified
- `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` - 15-cell Jupyter notebook covering RMSNorm, sinusoidal_embedding_1d, modulate from diffsynth/models/wan_video_dit.py

## Decisions Made
- Used upward directory search (`for _candidate in [_here, _here.parent, ...]`) rather than fixed `pathlib.Path("..").resolve()` — the fixed path fails in git worktrees where `Course/` is several levels deep inside `.claude/worktrees/`. The upward search handles both normal project checkout and worktree scenarios, and is more robust to future layout changes.
- Did not create a separate Task 1 commit since git doesn't track empty directories; the `Course/` directory creation is implicit in the notebook file commit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed PROJECT_ROOT path resolution for git worktree execution**
- **Found during:** Task 2 (NB-01 notebook execution verification)
- **Issue:** `pathlib.Path("..").resolve()` returned the worktree root (`.claude/worktrees/agent-.../`), not the project root containing `diffsynth/`. When run from `Course/` inside the worktree, `diffsynth/models/wan_video_dit.py` was not found, causing `FileNotFoundError`.
- **Fix:** Replaced fixed `..` path with an upward search checking `_here`, `_here.parent`, and `_here.parent.parent` for the presence of `diffsynth/models/wan_video_dit.py`. This works correctly in both normal checkout (`Course/../diffsynth/`) and the worktree scenario. For the final delivered notebook (committed to `Course/` at project root), `_here.parent` resolves correctly.
- **Files modified:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` (setup cell, Cell 2)
- **Verification:** Notebook executed from main project `Course/` directory in 3.5 seconds with all cells passing.
- **Committed in:** `9acc7cd` (same as task commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug)
**Impact on plan:** Fix is essential for portability. No scope creep. The delivered notebook works correctly in its final location (`Course/` at project root).

## Issues Encountered
- nbconvert notebook execution must be run from the notebook's directory (or a directory where the path search resolves correctly). Verified by running from main project's `Course/` directory.

## User Setup Required
None - no external service configuration required. Jupyter is installed and notebooks run CPU-only.

## Next Phase Readiness
- NB-01 is complete and sets the template for NB-02 through NB-05
- Cell 2 (Setup) can be copied verbatim to all subsequent notebooks (change only the symbols imported at the end)
- All patterns established: STD-01 through STD-07, D-01 through D-04
- Plan 01-02 (QKV projections, multi-head layout) can proceed immediately

---
*Phase: 01-dit-foundations*
*Completed: 2026-04-24*
