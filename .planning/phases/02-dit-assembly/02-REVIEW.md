---
phase: 02-dit-assembly
reviewed: 2026-04-24T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - Course/NB-06-dit-block.ipynb
  - Course/NB-07-patchify-unpatchify.ipynb
  - Course/NB-08-wanmodel-forward.ipynb
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
severity_max: medium
---

# Phase 02: Code Review Report -- DiT Assembly Notebooks

**Reviewed:** 2026-04-24
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three educational Jupyter notebooks were reviewed against `diffsynth/models/wan_video_dit.py` as the ground-truth source. All assertions are correct and would pass on CPU execution. All shape traces are accurate. The importlib path-search setup cell is robust (10-level upward walk, filesystem-root guard). No security, injection, or runtime-crash issues were found.

Three issues were identified:

1. **Medium (Warning):** NB-06 Exercise 3 teaches an incorrect formula for LoRA parameter counts on rectangular weight matrices. This is educational misinformation that a reader would act on.
2. **Low (Info):** NB-06 summary table cites `GateModule.forward` at line 190 (the class definition line) instead of line 194 (the `forward` method line).
3. **Low (Info):** NB-08 cites `WanModel.__init__` as lines 274-328, but the method body continues through line 338 (the `control_adapter` else-branch), omitting 10 lines of initialization.

---

## Warnings

### WR-01: Incorrect LoRA Parameter Formula for Rectangular Matrices in Exercise 3

**File:** `Course/NB-06-dit-block.ipynb` (Exercise 3, final markdown cell `nb06-cell-15`)
**Issue:** Exercise 3 instructs readers to calculate LoRA adapter parameters for `ffn.0` (shape `1536 -> 8960`) using the formula `2 * max(1536, 8960) * r`. This formula is wrong. The correct formula for a LoRA adapter on a weight matrix of shape `(in_features, out_features)` is `r * (in_features + out_features)` -- one low-rank matrix A of shape `(r, in_features)` and one B of shape `(out_features, r)`.

For `ffn.0` (Linear `1536 -> 8960`) with `r=16`:
- Notebook formula: `2 * max(1536, 8960) * 16 = 2 * 8960 * 16 = 286,720` -- **incorrect**
- Correct formula: `16 * (1536 + 8960) = 16 * 10,496 = 167,936`

The error overstates the adapter parameter count by 71%. A reader who applies this formula to other rectangular LoRA targets (e.g. `ffn.2`: `8960 -> 1536`) will get incorrect results and may draw wrong conclusions about LoRA compression ratios.

The formula for the square attention matrices (`2 * dim * r = 2 * 1536 * 16 = 49,152`) is correct in the same exercise because `in == out == 1536`, making `max` and `in+out` equivalent up to the factor-of-2 coincidence.

**Fix:** Replace the formula description in Exercise 3 with:

```
For ffn.0 (1536->8960): LoRA uses A of shape (r, in_features) and B of shape (out_features, r)
Total = r * (in_features + out_features) = 16 * (1536 + 8960) = 16 * 10,496 = 167,936 parameters

For ffn.2 (8960->1536): Total = 16 * (8960 + 1536) = 16 * 10,496 = 167,936 parameters

Compression ratios:
  ffn.0 original: 13,771,520 params;  LoRA r=16: 167,936 params  (82x compression)
  ffn.2 original: 13,764,096 params;  LoRA r=16: 167,936 params  (82x compression)
```

---

## Info

### IN-01: GateModule.forward Source Citation Points to Class Line, Not Method Line

**File:** `Course/NB-06-dit-block.ipynb` (Summary table, `nb06-cell-14`)
**Issue:** The Summary's source reference table lists `GateModule.forward` at line 190. Line 190 is the `class GateModule(nn.Module):` declaration. The `def forward(self, x, gate, residual):` is at line 194.
**Fix:** Change the table entry from line `190` to line `194`.

```markdown
| `GateModule.forward` | `diffsynth/models/wan_video_dit.py`, line 194 |
```

---

### IN-02: WanModel.__init__ Source Citation Truncated by 10 Lines

**File:** `Course/NB-08-wanmodel-forward.ipynb` (Cell `nb08-cell-06`, comment and cell `nb08-cell-15` summary table)
**Issue:** Both the code comment `# Source: diffsynth/models/wan_video_dit.py, lines 273-328` and the summary table entry `WanModel.__init__ | line 274` imply the `__init__` body ends at line 328. The body actually ends at line 338 -- lines 329-338 contain the conditional `has_image_input`, `has_ref_conv`, and `control_adapter` initialization logic. While these are optional features not used in the `has_image_input=False` demo, a reader checking the cited range will not find the `control_adapter` and `img_emb` setup.

**Fix:** Update the citation ranges to `274-338`:

In `nb08-cell-06` code comment:
```python
# Source: diffsynth/models/wan_video_dit.py, lines 274-338
```

In `nb08-cell-15` summary table:
```markdown
| `WanModel.__init__` | `diffsynth/models/wan_video_dit.py`, lines 274-338 |
```

---

## Verified Correct

The following were explicitly checked and found accurate:

| Check | Result |
|-------|--------|
| freqs shape assertion: `[F*H*W, 1, head_dim//2]` = `[64, 1, 64]` complex64 | Correct (22+21+21=64 complex pairs) |
| `precompute_freqs_cis_3d(128)` band sizes: f=22, h=21, w=21 | Correct (44//2=22, 42//2=21) |
| LoRA parameter count per block: 46,422,272 | Correct (verified arithmetic) |
| 30-block LoRA total: 1,392,668,160 | Correct (46,422,272 * 30) |
| `WanModel.forward` passes `t` (not `t_mod`) to `Head` | Correct; documented in NB-07 |
| `Head.forward` branching: `t` shape `[B, dim]` -> `else` branch | Correct for B=1 demo |
| Unpatchify rearrange string matches source exactly | Correct (line 353-354) |
| All shape trace entries in NB-08 Step table (Steps 1-8) | Correct |
| Gradient checkpointing condition `self.training and use_gradient_checkpointing` | Correct; line 393 |
| `x = x.requires_grad_(True)` at line 394 before block loop | Correctly cited |
| 48-channel concat: `torch.cat([noise, control, ref], dim=1)` | Correct |
| DiTBlock context shape: `[B, 277, dim]` = 257 CLIP + 20 text for `has_image_input=True` | Correct |
| `time_projection` output: `[B, 9216]` -> `unflatten(1, (6, 1536))` -> `[B, 6, 1536]` | Correct |
| NB-06 annotated cell chunk dimension `dim=1` for `t_mod` shape `[B, 6, dim]` | Correct (has_seq=False path) |
| NB-07 patchify assertions (cell 4): `x_seq.shape == [B, f*h*w, dim]` | Correct |
| NB-07 unpatchify assertion (cell 6): `x_video.shape == [B, out_dim, F, H, W]` | Correct |
| All other source line citations (20 of 22 checked) | Correct |

---

_Reviewed: 2026-04-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
