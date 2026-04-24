---
phase: 01-dit-foundations
reviewed: 2026-04-24T14:30:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb
  - Course/NB-02-qkv-projections-head-layout.ipynb
  - Course/NB-03-3d-rope.ipynb
  - Course/NB-04-self-cross-attention.ipynb
  - Course/NB-05-adaln-zero-modulation.ipynb
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-24T14:30:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Five Jupyter notebooks covering the foundational building blocks of the Wan2.1 DiT architecture were reviewed. The notebooks are well-structured educationally, with clear learning objectives, accurate shape assertions, and progressively building concepts. Code cells execute correctly and assertions are well-placed.

Four warnings were identified: one factual error in NB-01 claiming RMSNorm is used for DiT block pre-norms (the source code uses LayerNorm), one incorrect dtype claim in NB-03, one misleading exercise in NB-03, and an inconsistent project-root discovery pattern across notebooks. Three info-level items were also found.

No security issues (no hardcoded paths outside the project, no network calls, no credential exposure). The importlib setup pattern is robust and correctly stubs the camera controller dependency.

## Warnings

### WR-01: Factual Error — NB-01 Claims RMSNorm Used for DiT Block Pre-Norms

**File:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` (cell `712ac433`, Concept Map section)
**Issue:** The Concept Map states: "RMSNorm -> used in ... DiT block pre-norms (NB-06)". This is incorrect. The actual DiT block pre-norms (`norm1` and `norm2`) use `nn.LayerNorm(dim, eps=eps, elementwise_affine=False)` (source: `wan_video_dit.py` lines 207-208). RMSNorm is used only for q/k normalization inside SelfAttention and CrossAttention (`norm_q`, `norm_k`, `norm_k_img`). This will confuse students who read NB-01's concept map and then see LayerNorm in the actual DiTBlock code.
**Fix:** Change the Concept Map entry to:
```
RMSNorm -> used in SelfAttention and CrossAttention for q/k normalization (NB-04)
```
Remove the "DiT block pre-norms (NB-06)" claim entirely, or add a separate note that DiTBlock pre-norms use `nn.LayerNorm` with `elementwise_affine=False`.

### WR-02: Incorrect Dtype Claim — NB-03 Says precompute_freqs_cis Returns complex64

**File:** `Course/NB-03-3d-rope.ipynb` (cell `cell-nb03-03`, markdown)
**Issue:** The markdown states: "The returned tensor is `complex64` (single-precision complex), which is sufficient for the downstream multiply in rope_apply." This is incorrect. `precompute_freqs_cis` uses `.double()` (float64) for frequency computation and calls `torch.polar(torch.ones_like(freqs), freqs)` where both arguments are float64 tensors. `torch.polar` with float64 inputs returns **complex128** (double-precision complex), not complex64. The cell `cell-nb03-04` prints the actual dtype, which will show `torch.complex128` at runtime -- contradicting the markdown above it.
**Fix:** Change the description to:
```
The returned tensor is `complex128` (double-precision complex) because `torch.polar` is called
with float64 inputs. In `rope_apply`, the input is also upcast to float64 before
`view_as_complex`, so the complex multiply is done at full double precision.
```

### WR-03: Misleading Exercise — NB-03 Exercise 3 Implies view_as_complex Requires float64

**File:** `Course/NB-03-3d-rope.ipynb` (cell `cell-nb03-13`, Exercise 3)
**Issue:** Exercise 3 instructs students to "try: `torch.view_as_complex(torch.randn(2, 2))`" and asks "What error do you get?" followed by "Why does PyTorch require float64 (not float32) for `view_as_complex` when used with `torch.polar` output?" In reality, `torch.view_as_complex(torch.randn(2, 2))` works without error -- it takes float32 input and returns complex64. PyTorch does NOT require float64 for `view_as_complex`. The reason `rope_apply` upcasts to float64 is that the precomputed `freqs` tensor is complex128 (from float64 polar computation), so the input must also be complex128 for the multiplication to work without truncation. Students who follow this exercise will get no error and be confused.
**Fix:** Rewrite Exercise 3:
```
### Exercise 3 -- Why float64 in rope_apply?
In `rope_apply`, the input is upcast to float64 before `view_as_complex`. Try creating two
complex tensors at different precisions and multiplying them:
    a = torch.view_as_complex(torch.randn(2, 2))            # complex64
    b = torch.view_as_complex(torch.randn(2, 2).double())   # complex128
    print(a.dtype, b.dtype)
    c = a * b   # What dtype is c?
Why does the codebase choose to match the float64 precision of the freqs tensor
rather than downcast freqs to complex64?
```

### WR-04: Inconsistent Project Root Discovery Across Notebooks

**File:** `Course/NB-01-rmsnorm-sinusoidal-modulate.ipynb` (cell `c07635b0`), `Course/NB-04-self-cross-attention.ipynb` (cell `nb04-cell-02`), `Course/NB-05-adaln-zero-modulation.ipynb` (cell `nb05-cell-02`)
**Issue:** NB-02 and NB-03 use the more robust traversal `[_here] + list(_here.parents)[:6]` (up to 7 levels), while NB-01, NB-04, and NB-05 use only `[_here, _here.parent, _here.parent.parent]` (3 levels). The setup comment in NB-01/04/05 says "This handles both normal checkout and git worktree scenarios" but 3 levels may not be sufficient for deeper worktree paths. This inconsistency means NB-01/04/05 could fail to find the project root in the same worktree setup where NB-02/03 succeed.
**Fix:** Standardize all five notebooks to use the same traversal as NB-02/03:
```python
for _candidate in [_here] + list(_here.parents)[:6]:
    if (_candidate / "diffsynth" / "models" / "wan_video_dit.py").exists():
        PROJECT_ROOT = _candidate
        break
```

## Info

### IN-01: NB-03 Markdown Overstates torch.polar Requirement

**File:** `Course/NB-03-3d-rope.ipynb` (cell `cell-nb03-03`)
**Issue:** The markdown states "`torch.polar` requires float64 (double precision) inputs." This is technically false -- `torch.polar` accepts float32 inputs and returns complex64. The Wan2.1 code *chooses* to use float64 for numerical precision, but it is not a PyTorch API requirement. Students may form an incorrect mental model of the API.
**Fix:** Change to: "`torch.polar` is called with float64 inputs for precision -- this is a design choice for numerical accuracy, not a PyTorch API requirement."

### IN-02: NB-04 Does Not Mention AttentionModule Wrapper

**File:** `Course/NB-04-self-cross-attention.ipynb` (cell `nb04-cell-03`, markdown)
**Issue:** The architecture walkthrough lists "Four linear projections" and "RMSNorm" components but does not mention the `AttentionModule` wrapper (line 115-122 in source), which wraps `flash_attention` into an `nn.Module`. This is a thin wrapper (just calls `flash_attention`), but it appears in the `named_modules()` output printed in cell `nb04-cell-04` and students may wonder what it is.
**Fix:** Add a brief note: "The `AttentionModule` is a thin `nn.Module` wrapper around `flash_attention` -- it has no learnable parameters and simply delegates to the backend-selection logic described in section 5 of NB-02."

### IN-03: NB-05 Only Covers 3D t_mod Shape

**File:** `Course/NB-05-adaln-zero-modulation.ipynb` (cell `nb05-cell-07`)
**Issue:** The notebook demonstrates the six-parameter chunk operation with `combined.chunk(6, dim=1)`, which assumes `t_mod` shape `[B, 6, dim]`. The actual DiTBlock.forward (lines 216-217) has a branch for 4D `t_mod` (`has_seq = len(t_mod.shape) == 4`; `chunk_dim = 2 if has_seq else 1`). This 4D path is not mentioned. For NB-05's scope this is acceptable, but a brief note would help students who later read the source.
**Fix:** Add a note in the markdown: "Note: DiTBlock.forward also handles a 4D `t_mod` shape (when per-token conditioning is used). In that case, the chunk happens on `dim=2` instead of `dim=1`, and the six parameters are squeezed to remove the extra dimension. This path is covered in NB-06."

---

_Reviewed: 2026-04-24T14:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
