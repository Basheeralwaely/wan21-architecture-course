---
phase: 03-vae-track
reviewed: 2026-04-24T14:22:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - Course/NB-09-causalconv3d-resblock-attn.ipynb
  - Course/NB-10-vae-encoder.ipynb
  - Course/NB-11-vae-decoder-patchify.ipynb
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-04-24T14:22:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three Jupyter notebooks (NB-09, NB-10, NB-11) forming the VAE track of an educational course on the Wan video generation architecture were reviewed. These notebooks are well-structured educational material that teaches CausalConv3d, ResidualBlock, AttentionBlock, Encoder3d, Decoder3d, reparameterization, and patchify/unpatchify.

All source line references were verified against `diffsynth/models/wan_video_vae.py` and are accurate. The production configuration values (dim=96, z_dim=16, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True]) match the `VideoVAE_` defaults in the source. Shape tables, architecture descriptions, and code demonstrations are correct.

Two warnings were found related to incorrect labeling in an ASCII diagram and a misleading `F.pad` tuple description. Three informational items were noted regarding minor improvements to educational clarity. No security issues, no bugs in executable code, no hardcoded secrets.

## Warnings

### WR-01: Incorrect ASCII diagram in CausalConv3d temporal padding explanation

**File:** `Course/NB-09-causalconv3d-resblock-attn.ipynb` (cell `nb09-cell-03`, markdown)
**Issue:** The ASCII diagram shows two convolution windows for a T=3 input, but with kernel_t=3 and 2 zeros prepended (total T=5 after padding), there should be exactly 3 valid windows. The second window shown -- `[window-1] = uses f-1, f, 0 <- right pad = 0` -- is incorrect. It implies a window that extends past the right edge and uses zero-padded values, but since `right_T=0` (no right temporal padding), this window does not exist. The three actual windows are:
- Window 0: `[0, 0, f-2]` -> output for frame 0
- Window 1: `[0, f-2, f-1]` -> output for frame 1
- Window 2: `[f-2, f-1, f]` -> output for frame 2

**Fix:** Replace the convolution window portion of the ASCII diagram:
```
After F.pad:     [  0] [  0] [f-2] [f-1] [ f ]   <- T=5 after padding

Conv3d k=3:   [----window-0----]                   = uses 0, 0, f-2   -> output frame 0
                    [----window-1----]              = uses 0, f-2, f-1 -> output frame 1
                          [----window-2----]        = uses f-2, f-1, f -> output frame 2 (no future!)
Output T = 5 - 3 + 1 = 3 = Input T (shape preserved, causal: no future frames used)
```

### WR-02: Misleading F.pad tuple element labels in CausalConv3d description

**File:** `Course/NB-09-causalconv3d-resblock-attn.ipynb` (cell `nb09-cell-02`, markdown, and cell `nb09-cell-03`, markdown)
**Issue:** The `_padding` tuple is labeled as `(right_W, right_W, bot_H, bot_H, 2*t_pad, 0)`. The correct F.pad semantics for a 5D tensor are `(left_W, right_W, top_H, bot_H, front_T, back_T)`. Since `CausalConv3d` uses symmetric padding for W and H (same value on both sides), the numerical result is correct, but the labels `right_W, right_W` and `bot_H, bot_H` are wrong -- they should be `left_W, right_W` and `top_H, bot_H`. For educational material teaching padding semantics, incorrect labels undermine the learning objective.
**Fix:** Change the label in cell nb09-cell-02 and nb09-cell-03 from:
```
_padding = (right_W, right_W, bot_H, bot_H, 2*t_pad, 0)
```
to:
```
_padding = (left_W, right_W, top_H, bot_H, 2*t_pad, 0)
```
And update cell nb09-cell-04 comment accordingly:
```python
# F.pad order: (left_W, right_W, top_H, bot_H, front_T, back_T)
```
(This comment is already correct in cell 04 -- the inconsistency is between cells.)

## Info

### IN-01: Typo in source code docstring reference

**File:** `Course/NB-09-causalconv3d-resblock-attn.ipynb` (cell `nb09-cell-00`, markdown)
**Issue:** The comment `(right_W, right_W, bot_H, bot_H, 2*t_pad, 0)` in the Learning Objectives section repeats the same labeling error as WR-02. While the Learning Objectives are a summary, they should be consistent with the correct explanation below.
**Fix:** Update to `(left_W, right_W, top_H, bot_H, 2*t_pad, 0)`.

### IN-02: NB-10 concept map shows Level 1 temporal compression annotation that may confuse readers

**File:** `Course/NB-10-vae-encoder.ipynb` (cell `nb10-cell-00`, markdown)
**Issue:** The concept map annotation on Level 1 says `(temporal unchanged w/o feat_cache)` but Level 0 (which also preserves temporal) does not have this annotation. This asymmetry could lead readers to think Level 0 does change temporal dimensions. Adding a consistent note or removing the Level 1 annotation would improve clarity.
**Fix:** Either add `(temporal always unchanged)` to Level 0 as well, or move the annotation to a note below the diagram that applies to all levels.

### IN-03: NB-10 encoder default parameter mismatch with source could cause confusion

**File:** `Course/NB-10-vae-encoder.ipynb` (cell `nb10-cell-03`, code)
**Issue:** The notebook uses `temperal_downsample=[False, True, True]` which matches the `VideoVAE_` production default, but differs from `Encoder3d`'s own default of `[True, True, False]` (source line 525). While the notebook correctly notes "SAME as production" and the production config annotation below confirms this, a reader looking at `Encoder3d`'s signature directly might be confused. A brief comment noting the difference between class default and production usage would help.
**Fix:** Add a comment in the demo config block:
```python
# Note: Encoder3d's default is [True, True, False], but VideoVAE_ overrides
# this to [False, True, True] -- we use the production value here.
```

---

_Reviewed: 2026-04-24T14:22:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
