---
phase: 01-dit-foundations
plan: 03
subsystem: notebooks
tags: [jupyter, pytorch, self-attention, cross-attention, rope, adaln-zero, modulation, diffsynth, importlib, nbformat]

# Dependency graph
requires:
  - "01-01 (NB-01 template, importlib pattern, STD-01 through STD-07)"
provides:
  - "Course/NB-04-self-cross-attention.ipynb — SelfAttention and CrossAttention walkthrough"
  - "Course/NB-05-adaln-zero-modulation.ipynb — adaLN-Zero modulation and per-block offset walkthrough"
affects: [later-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "3D freqs assembly (F=S, H=1, W=1) as pedagogical shortcut for rope_apply demo in NB-04"
    - "gate=0 identity numerical assertion: assert torch.allclose(out_zero, x)"
    - "Six-parameter chunk from modulation: combined.chunk(6, dim=1)"

key-files:
  created:
    - "Course/NB-04-self-cross-attention.ipynb"
    - "Course/NB-05-adaln-zero-modulation.ipynb"
  modified: []

key-decisions:
  - "Used full 3D freqs assembly (F=S, H=1, W=1) instead of simplified f_freqs[:S].unsqueeze(1) — the simplified approach produces [S,1,22] which mismatches rope_apply's expected [S,1,64]; full assembly is correct and mirrors production code"
  - "NB-05 demonstrates gate=0 identity by manual construction (not via DiTBlock.forward) — avoids DiTBlock's freqs and context dependencies, keeping the demo focused on the modulation concept"

patterns-established:
  - "freqs assembly for rope_apply demos: use full 3D grid with F=seq_len, H=1, W=1 to get [S, 1, head_dim//2]"
  - "Pitfall 4 pattern: show gate=0 identity first (constructed), then separately explain random init — do not conflate"

requirements-completed: [DIT-08, DIT-09, DIT-10, DIT-11]

# Metrics
duration: 8min
completed: 2026-04-24
---

# Phase 1 Plan 03: DiT Foundations NB-04 and NB-05 Summary

**NB-04 walks the full SelfAttention forward pass (q/k RMSNorm+RoPE, v unnormalized) and CrossAttention dual-stream CLIP+T5 split; NB-05 demonstrates adaLN-Zero six-parameter chunk, gate=0 identity, and Wan2.1's random-init modulation vs DiT paper zero-init — both notebooks run CPU-only under 5 seconds**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-24T11:39:44Z
- **Completed:** 2026-04-24T11:47:24Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `Course/NB-04-self-cross-attention.ipynb` (16 cells) covering DIT-08 (SelfAttention with RoPE integration) and DIT-09 (CrossAttention dual-stream CLIP/T5 path)
- Created `Course/NB-05-adaln-zero-modulation.ipynb` (14 cells) covering DIT-10 (adaLN-Zero gate=0 identity, six-parameter chunk) and DIT-11 (per-block modulation learned offset)
- NB-04 executes on CPU in ~4.7 seconds; NB-05 in ~3.6 seconds — both within the 5-second budget
- All 7 STD standards followed in both notebooks; 3 modification exercises each (D-03/D-04)
- Pitfall 4 explicitly addressed in NB-05: Wan2.1 uses `torch.randn(1,6,dim)/dim**0.5`, not zero-init
- Pitfall 5 addressed in NB-04: CrossAttention `y[:,:257]` = CLIP image tokens, `y[:,257:]` = T5 text

## Task Commits

1. **Task 1: Write NB-04** — `4de3c13` (feat)
2. **Task 2: Write NB-05** — `92c2d5a` (feat)

## Files Created/Modified

- `Course/NB-04-self-cross-attention.ipynb` — 16-cell notebook: SelfAttention (norm_q/norm_k asymmetry, freqs assembly, forward pass), CrossAttention (has_image_input=True dual-stream, parameter comparison)
- `Course/NB-05-adaln-zero-modulation.ipynb` — 14-cell notebook: GateModule gate=0 identity, six-parameter chunk, Wan2.1 vs DiT paper init, modulate connection (NB-01), per-block specialization demo

## Decisions Made

- Used full 3D freqs assembly (`F=S, H=1, W=1`) for NB-04 SelfAttention demo instead of the simplified `f_freqs[:S].unsqueeze(1)` approach in PATTERNS.md. The simplified approach gives `[S, 1, 22]` but `rope_apply` requires `[S, 1, head_dim//2]` = `[S, 1, 64]`. The full assembly correctly concatenates all three frequency bands and mirrors the production code from NB-03.
- NB-05 demonstrates the GateModule identity property via manual tensor construction (`x + gate_zero * residual`) rather than running `DiTBlock.forward`. This avoids the need for `freqs` and `context` inputs, keeping the cell focused on the gate=0 concept.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed freqs shape mismatch in SelfAttention demo**
- **Found during:** Task 1 (NB-04 execution verification)
- **Issue:** PATTERNS.md and RESEARCH.md show `freqs = f_freqs[:S].unsqueeze(1)` (shape `[S, 1, 22]`), but `rope_apply` in `wan_video_dit.py` expects `freqs` of shape `[S, 1, head_dim//2]` = `[S, 1, 64]`. The temporal-only band only has 22 complex numbers (f_dim//2), not the full 64 needed.
- **Fix:** Replaced simplified freqs with the full 3D grid assembly: `torch.cat([f_freqs[:F]..., h_freqs[:H]..., w_freqs[:W]...], dim=-1).reshape(FHW, 1, -1)` with `F=S, H=1, W=1`. This produces the correct `[S, 1, 64]` shape and mirrors the production code shown in NB-03.
- **Files modified:** `Course/NB-04-self-cross-attention.ipynb` (cell 6)
- **Committed in:** `4de3c13`

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug)
**Impact on plan:** Fix is required for correct execution. The assembly approach is actually pedagogically better — it shows the same pattern as NB-03's production grid assembly, reinforcing that connection.

## Issues Encountered

- None beyond the freqs shape mismatch (auto-fixed above)

## User Setup Required

None — no external service configuration required. Both notebooks run CPU-only with the existing environment.

## Known Stubs

None — all cells use real diffsynth symbols and produce real tensor outputs.

## Threat Flags

None — notebooks contain no network calls, no secrets, no user data, no file writes beyond the notebook itself.

## Self-Check

Verified:
- `Course/NB-04-self-cross-attention.ipynb` — exists, 16 cells, all structure checks passed, executes in 4.7s
- `Course/NB-05-adaln-zero-modulation.ipynb` — exists, 14 cells, all structure checks passed, executes in 3.6s
- Commit `4de3c13` — NB-04 feat commit
- Commit `92c2d5a` — NB-05 feat commit

## Self-Check: PASSED

---
*Phase: 01-dit-foundations*
*Completed: 2026-04-24*
