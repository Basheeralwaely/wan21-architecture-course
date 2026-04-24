---
phase: 02-dit-assembly
fixed_at: 2026-04-24T00:00:00Z
review_path: .planning/phases/02-dit-assembly/02-REVIEW.md
iteration: 1
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-04-24T00:00:00Z
**Source review:** .planning/phases/02-dit-assembly/02-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 1
- Fixed: 1
- Skipped: 0

## Fixed Issues

### WR-01: Incorrect LoRA Parameter Formula for Rectangular Matrices in Exercise 3

**Files modified:** `Course/NB-06-dit-block.ipynb`
**Commit:** 9193813
**Applied fix:** Replaced the incorrect formula `2 * max(in, out) * r` with the correct `r * (in_features + out_features)` in Exercise 3's markdown cell (nb06-cell-15). Updated the low-rank matrix shape descriptions from `A(dim, r)` / `B(r, dim)` to the general `A(r, in_features)` / `B(out_features, r)`. Corrected the ffn.0 adapter size from the implied 286,720 (via the wrong formula) to the correct 167,936 = 16 * (1536 + 8960). The square-matrix case (self_attn.q) was also clarified with `16 * (1536 + 1536) = 49,152` to make the general formula visible, though the numerical result was already correct.

## Out-of-Scope Findings

The following Info-level findings were not in scope for this fix run (`fix_scope: critical_warning`):

### IN-01: GateModule.forward Source Citation Points to Class Line, Not Method Line

**File:** `Course/NB-06-dit-block.ipynb` (Summary table, nb06-cell-14)
**Reason:** Out of scope (Info severity, fix_scope is critical_warning)
**Original issue:** Summary table cites `GateModule.forward` at line 190 (class definition) instead of line 194 (forward method).

### IN-02: WanModel.__init__ Source Citation Truncated by 10 Lines

**File:** `Course/NB-08-wanmodel-forward.ipynb` (nb08-cell-06, nb08-cell-15)
**Reason:** Out of scope (Info severity, fix_scope is critical_warning)
**Original issue:** Citation range for `WanModel.__init__` ends at line 328 but the method body continues through line 338, omitting the `control_adapter` and `img_emb` initialization.

---

_Fixed: 2026-04-24T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
