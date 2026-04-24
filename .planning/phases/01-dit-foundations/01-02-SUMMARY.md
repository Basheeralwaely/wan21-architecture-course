---
phase: 01-dit-foundations
plan: 02
subsystem: notebooks
tags: [jupyter, pytorch, qkv-projections, multi-head-attention, einops, 3d-rope, rotary-position-embedding, rope-apply, diffsynth, nbformat]

# Dependency graph
requires:
  - phase: 01-01
    provides: "NB-01 template (importlib setup, STD-01..STD-07 conventions, D-01..D-04 exercise format)"
provides:
  - "Course/NB-02-qkv-projections-head-layout.ipynb — QKV projection and multi-head layout walkthrough"
  - "Course/NB-03-3d-rope.ipynb — 3D RoPE frequency precomputation walkthrough"
  - "Upward path search pattern (6 levels) verified for git worktree execution"
affects: [01-03, later-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Upward parent search using `[_here] + list(_here.parents)[:6]` — covers both normal checkout and deep git worktree paths"
    - "rope_apply requires full [S, 1, head_dim//2] assembled freqs — NOT a single-axis band"
    - "3D grid assembly: broadcast each axis table across F/H/W, cat on last dim, reshape to [seq_len, 1, head_dim//2]"

key-files:
  created:
    - "Course/NB-02-qkv-projections-head-layout.ipynb"
    - "Course/NB-03-3d-rope.ipynb"
  modified: []

key-decisions:
  - "Extended path search from 3 levels (NB-01) to 6 levels — worktree Course/ is 4 levels below project root, requiring the extra depth"
  - "rope_apply demo uses fully assembled [S, 1, head_dim//2] freqs from the 3D grid, not a single-axis band — single-axis causes a shape mismatch at the complex multiply"

patterns-established:
  - "STD-07 setup cell with 6-level upward search: copy to NB-04 and NB-05"
  - "3D frequency grid assembly pattern: f_freqs[:F].view+expand, h_freqs[:H].view+expand, w_freqs[:W].view+expand, cat, reshape"

requirements-completed: [DIT-04, DIT-05, DIT-06, DIT-07]

# Metrics
duration: 8min
completed: 2026-04-24
---

# Phase 1 Plan 02: DiT Foundations NB-02 and NB-03 Summary

**QKV projection/multi-head layout (NB-02, 10 cells) and 3D RoPE frequency precomputation (NB-03, 14 cells) — both execute CPU-only in under 5 seconds with all assertions passing**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-24T11:39:25Z
- **Completed:** 2026-04-24T11:47:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `Course/NB-02-qkv-projections-head-layout.ipynb` (10 cells) covering DIT-04 (dim-to-dim QKV projections from SelfAttention line 125) and DIT-05 (dual head layout conventions (B,S,N,D) vs (B,N,S,D) from flash_attention line 28), with round-trip lossless reshape verification
- Created `Course/NB-03-3d-rope.ipynb` (14 cells) covering DIT-06 (three-axis band split: f_dim=44, h_dim=42, w_dim=42 for head_dim=128) and DIT-07 (precompute_freqs_cis_3d with complex64 output, float64 requirement, 3D grid assembly, rope_apply dtype preservation)
- Extended the importlib setup cell path search to 6 levels — the NB-01 3-level search was insufficient for the 4-level-deep git worktree path

## Task Commits

Each task was committed atomically:

1. **Task 1: Write NB-02 — QKV projections and multi-head attention layout** - `cd481c6` (feat)
2. **Task 2: Write NB-03 — 3D RoPE frequency precomputation** - `240d4d5` (feat)

## Files Created/Modified
- `Course/NB-02-qkv-projections-head-layout.ipynb` - 10-cell notebook: QKV dim-to-dim projections, (B,S,N,D)/(B,N,S,D) dual layout demo, lossless round-trip verification, 3 exercises
- `Course/NB-03-3d-rope.ipynb` - 14-cell notebook: 1D RoPE base function, three-axis band arithmetic (f/h/w split), precompute_freqs_cis_3d, 3D grid assembly, rope_apply dtype handling, 3 exercises

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended upward path search from 3 to 6 levels for git worktree compatibility**
- **Found during:** Task 1 (NB-02 notebook execution verification)
- **Issue:** NB-01 used `[_here, _here.parent, _here.parent.parent]` (3 candidates). In the git worktree, `Course/` is at `.claude/worktrees/agent-.../Course/` — 4 levels below the project root containing `diffsynth/`. The 3-level search failed with `FileNotFoundError`.
- **Fix:** Changed to `[_here] + list(_here.parents)[:6]` — searches up to 6 levels, finding `diffsynth/` at level 4 in the worktree and level 1 in the final project layout.
- **Files modified:** NB-02 setup cell, NB-03 setup cell (same fix applied to both)
- **Committed in:** `cd481c6`, `240d4d5`

**2. [Rule 1 - Bug] Fixed rope_apply freqs shape in NB-03 demo cell**
- **Found during:** Task 2 (NB-03 notebook execution verification)
- **Issue:** The plan's PATTERNS.md example used `f_freqs_local[:S].unsqueeze(1)` — a single-axis band of shape `[S, 1, f_dim//2]` = `[20, 1, 22]`. `rope_apply` internally reshapes `x` to `[B, S, N, head_dim]` then views as complex `[B, S, N, head_dim//2]`, requiring freqs of size `head_dim//2 = 64`, not `f_dim//2 = 22`. Shape mismatch caused `RuntimeError: The size of tensor a (64) must match the size of tensor b (22)`.
- **Fix:** Built the full 3D assembled freqs (`[S, 1, head_dim//2]`) using F_s=5, H_s=2, W_s=2 (so F*H*W=20=S) via the same broadcast+cat pattern shown in the grid assembly cell. Added prose note clarifying the shape requirement.
- **Files modified:** `Course/NB-03-3d-rope.ipynb` (Cell 11, rope_apply demo)
- **Committed in:** `240d4d5`

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in execution)
**Impact:** Both fixes are essential for correct execution. No scope changes. The delivered notebooks work correctly in the final `Course/` at project root (path search finds diffsynth/ at level 1 in normal checkout).

## Known Stubs
None — all cells use real diffsynth symbols with real tensor operations. No placeholder data or hardcoded empty values.

## Threat Flags
None — no new network endpoints, auth paths, or trust boundaries introduced. Notebooks are read-only educational files that load a local Python module.

## Self-Check: PASSED
- `Course/NB-02-qkv-projections-head-layout.ipynb` — FOUND
- `Course/NB-03-3d-rope.ipynb` — FOUND
- `cd481c6` — FOUND (feat(01-02): add NB-02)
- `240d4d5` — FOUND (feat(01-02): add NB-03)

---
*Phase: 01-dit-foundations*
*Completed: 2026-04-24*
