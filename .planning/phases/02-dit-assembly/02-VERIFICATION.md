---
phase: 02-dit-assembly
verified: 2026-04-24T16:00:00Z
status: human_needed
score: 8/8 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Read Exercise 3 in NB-06 and apply the LoRA adapter size formula to ffn.0 (1536->8960) — confirm whether the formula 2*max(1536,8960)*r is presented and whether you accept this as a tolerable error in an exercise hint"
    expected: "The formula should be r*(in+out) = 16*(1536+8960) = 167,936. The notebook uses 2*max(1536,8960)*r = 286,720 (overstated by 71%). Decision needed: fix or accept."
    why_human: "WR-01 from code review — the formula is factually wrong for rectangular matrices. It does not affect any runnable assertion (it is prose in an exercise), but it teaches incorrect LoRA math. Whether to fix or accept is a content quality call."
  - test: "Confirm NB-06 summary table GateModule.forward citation (check whether line 190 or 194 is cited)"
    expected: "Citation should point to line 194 (the def forward method) not line 190 (the class definition)"
    why_human: "IN-01 from code review — minor citation accuracy issue. Fix takes 2 characters but requires author decision on whether to update."
  - test: "Confirm NB-08 WanModel.__init__ source citation range (cells nb08-cell-05 and nb08-cell-14 summary table)"
    expected: "Citation should read lines 274-338 (not 274-328) to include the has_image_input/has_ref_conv/control_adapter initialization"
    why_human: "IN-02 from code review — the actual __init__ method body continues through line 338. Fix is a range adjustment but requires author decision."
---

# Phase 02: DiT Assembly Verification Report

**Phase Goal:** Readers can trace a full DiT inference pass through a composed DiT Block, patch embedding/unpatchify, and the 30-block WanModel with 48-channel concatenated input
**Verified:** 2026-04-24T16:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

All automated checks pass. Three code-review findings require human decision before the phase can be marked fully closed. The findings do not affect notebook executability or the observable shape/assertion outcomes — they are educational accuracy issues in prose and exercise hints.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Reader can open NB-06 and see the DiTBlock's six sub-modules wired in `__init__`, trace the forward pass line-by-line with back-references to NB-01/NB-04/NB-05 | VERIFIED | 15 cells; self_attn, cross_attn, norm1/2/3, ffn, modulation, gate all present; "recall NB-01/04/05" back-references present; annotated 4-stage forward (shift_msa/gate_msa) in cell [9] |
| 2 | Reader can see per-sub-module parameter counts (q, k, v, o, ffn.0, ffn.2) with percentages, and understand why these are LoRA targets — then see the 30-block scaling | VERIFIED | Cells [11]+[12]: individual counts with % shown; "task-specific" rationale in prose; `num_layers = 30` and `1,392,668,160` 30-block total confirmed in output |
| 3 | Reader can open NB-07 and observe Conv3d patchify: (B,48,F,H,W) -> (B,F*(H//2)*(W//2),1536), then einops unpatchify reversing it back to (B,16,F,H,W) | VERIFIED | Cell [3]: Conv3d(48,1536,k=(1,2,2),s=(1,2,2)) output [1,1536,4,4,4] then rearranged to [1,64,1536]; Cell [5]: exact einops string `'b (f h w) (x y z c) -> b c (f x) (h y) (w z)'`; round-trip assertions pass |
| 4 | Reader can see the Head module applying 2-parameter adaLN (shift+scale) using raw time embedding t, not the 6-chunk t_mod | VERIFIED | NB-07 cell [9]: `head(x, t)` call (not t_mod); prose explicitly states "NOT the 6-parameter modulation from DiTBlock"; 2-parameter modulation `[1,2,1536]` printed |
| 5 | All cells in NB-06 run on CPU in under 5 seconds total with all assertions passing | VERIFIED | jupyter nbconvert exit 0; per-cell times: max 0.971s (setup), 0.209s (DiTBlock init), 0.025s (forward), 0.028s (annotated) — all well under 5s |
| 6 | Reader can see noise (16ch), control (16ch), and reference (16ch) latents created separately then concatenated into a 48-channel tensor via torch.cat on dim=1 | VERIFIED | NB-08 cell [3]: `noise_latent`, `control_latent`, `ref_latent` as separate randn tensors; `torch.cat([noise_latent, control_latent, ref_latent], dim=1)`; `assert x_48.shape == torch.Size([B, 48, F, H, W])` passes |
| 7 | Reader can trace WanModel.forward from 48-channel input through patchify, freqs assembly, 3 DiT blocks, Head, and unpatchify back to 16-channel output | VERIFIED | NB-08 cell [8]: 8-step shape trace comments; `model(x_48, timestep, context)` output `[1,16,4,8,8]`; assert passes; shape trace table in cell [9] markdown |
| 8 | Reader can see gradient checkpointing activate with model.train() + use_gradient_checkpointing=True and deactivate with model.eval() | VERIFIED | NB-08 cell [11]: `model.train()` before checkpoint call; `model.eval()` before inference; `assert out_gc.shape == out_eval.shape` passes; condition `self.training and use_gradient_checkpointing` explicitly documented |

**Score:** 8/8 truths verified

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| SC-1 | Reader can open NB-06 and see self-attn + cross-attn + FFN + adaLN composition in one forward pass, with a parameter count cell identifying LoRA target modules | VERIFIED | DiTBlock forward pass with all 4 stages present; LoRA target table with q/k/v/o/ffn.0/ffn.2 names and counts |
| SC-2 | Reader can open NB-07 and observe the patchify/unpatchify shape round-trip: (B,C,F,H,W) → flattened sequence → back to video dimensions | VERIFIED | Round-trip confirmed: [1,48,4,8,8] → [1,64,1536] → [1,16,4,8,8]; `assert x.shape[2:] == x_video.shape[2:]` passes |
| SC-3 | Reader can open NB-08 and run a full WanModel forward pass with dummy 48-channel input (noise + control + reference), seeing gradient checkpointing stripped for inference | VERIFIED | All three sub-goals met; gradient checkpoint train vs eval side-by-side confirmed |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Course/NB-06-dit-block.ipynb` | DiT Block composition walkthrough + LoRA parameter analysis | VERIFIED | 15 cells, valid JSON, executes exit 0, commits 5ee1380 |
| `Course/NB-07-patchify-unpatchify.ipynb` | Patchify/unpatchify round-trip + Head module walkthrough | VERIFIED | 13 cells, valid JSON, executes exit 0, commit cf3e52e |
| `Course/NB-08-wanmodel-forward.ipynb` | Full WanModel end-to-end forward pass with 48-channel input and gradient checkpointing | VERIFIED | 16 cells, valid JSON, executes exit 0, commit fd10d81 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| NB-06 | `diffsynth/models/wan_video_dit.py` | `from diffsynth.models.wan_video_dit import (DiTBlock, GateModule, modulate, precompute_freqs_cis_3d)` | WIRED | importlib 10-level upward walk; project root found at `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training`; classes instantiated and called successfully |
| NB-07 | `diffsynth/models/wan_video_dit.py` | `from diffsynth.models.wan_video_dit import WanModel, Head` | WIRED | WanModel and Head imported; Head instantiated and called with `head(x, t)` |
| NB-08 | `diffsynth/models/wan_video_dit.py` | `from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d` | WIRED | WanModel instantiated with 3 layers; full forward pass called with `model(x_48, timestep, context)` and gradient checkpointing path exercised |

### Data-Flow Trace (Level 4)

All notebooks use dummy `torch.randn()` tensors as input — no external data source is needed. Real `diffsynth` classes are instantiated and called (STD-07). Data flows from dummy input through real model code to verified output shapes. No hollow props or static returns.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| NB-06 | `out` from `block(x, context, t_mod, freqs)` | Real DiTBlock forward | Yes — output reflects actual weight computations | FLOWING |
| NB-07 | `x_video` from unpatchify rearrange | Real einops rearrange on actual Conv3d output | Yes — spatial round-trip confirmed by assertion | FLOWING |
| NB-08 | `out` from `model(x_48, timestep, context)` | Real WanModel forward (3 blocks) | Yes — output `[1,16,4,8,8]` reflects full 8-step pipeline | FLOWING |

### Behavioral Spot-Checks

| Behavior | Result | Status |
|----------|--------|--------|
| NB-06 executes without error | `jupyter nbconvert --execute` exit 0 | PASS |
| NB-07 executes without error | `jupyter nbconvert --execute` exit 0 | PASS |
| NB-08 executes without error | `jupyter nbconvert --execute` exit 0 | PASS |
| NB-06 DiTBlock output shape `[1, 64, 1536]` | Printed in executed output | PASS |
| NB-07 round-trip spatial dims match `(4, 8, 8)` | Printed in executed output | PASS |
| NB-08 WanModel output shape `[1, 16, 4, 8, 8]` | Printed in executed output | PASS |
| NB-08 gradient checkpoint paths both produce same shape | `assert out_gc.shape == out_eval.shape` passes | PASS |
| Per-cell timing under 5s (STD-03) | Max cell time 0.971s (setup cell with module loading); all compute cells under 0.83s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DIT-12 | 02-01-PLAN.md | NB-06 covers DiT Block — composition of self-attention + cross-attention + FFN + adaLN conditioning | SATISFIED | NB-06 cells [2]-[9]: all four components wired in __init__ and traced in forward |
| DIT-13 | 02-01-PLAN.md | NB-06 includes parameter count breakdown identifying LoRA target modules (q, k, v, o, ffn.0, ffn.2) | SATISFIED | NB-06 cells [11]+[12]: per-sub-module counts with percentages; 30-block total; WHY rationale in prose |
| DIT-14 | 02-01-PLAN.md | NB-07 covers patchify — Conv3d learned projection with patch size (1,2,2), video-to-token conversion | SATISFIED | NB-07 cell [3]: Conv3d(48,1536,k=(1,2,2)); output [1,64,1536] verified |
| DIT-15 | 02-01-PLAN.md | NB-07 covers unpatchify — token-to-video reconstruction, shape recovery | SATISFIED | NB-07 cell [5]: exact einops rearrange string; `assert x_video.shape == torch.Size([B, out_dim, F, H, W])` passes |
| DIT-16 | 02-02-PLAN.md | NB-08 covers full WanModel end-to-end — 30 blocks, 48-channel input composition (noise + control + reference) | SATISFIED | NB-08: 3-block demo runs; 30-block config annotated; 48-channel concat with all three latent types shown |
| DIT-17 | 02-02-PLAN.md | NB-08 covers gradient checkpointing stripping for inference context | SATISFIED | NB-08 cell [11]: train path with checkpointing vs eval path without; condition `self.training and use_gradient_checkpointing` documented |

No orphaned requirements — all 6 Phase 2 requirements (DIT-12 through DIT-17) are claimed by the two plans and satisfied by verified notebook content.

### User Decisions Coverage (from CONTEXT.md)

| Decision | Status | Evidence |
|----------|--------|----------|
| D-01: Bottom-up buildup in NB-06 (`__init__` then forward) | HONORED | NB-06 structure: sub-module inventory cell before forward walkthrough cells |
| D-02: ASCII block diagram in NB-06 | HONORED | Cell [0] markdown contains `adaLN`, `SelfAttention`, `CrossAttention`, `FFN`, `gate` |
| D-03: Counts + rationale, WHY explanation, no rank decomposition details | PARTIAL | Main prose (cells [10]-[12]) honors D-03 — counts, percentages, "task-specific" rationale. Exercise 3 introduces rank-r formula, which D-03 said to exclude. However exercise design is listed under "Claude's Discretion" in CONTEXT.md, and the PLAN spec itself included Exercise 3 |
| D-04: Per-block AND 30-block scope | HONORED | Cell [11] shows per-block breakdown; cell [12] scales to 30 blocks |
| D-05: ASCII spatial-to-sequence diagram in NB-07 | HONORED | NB-07 cell [0] markdown includes spatial-to-sequence ASCII with stride annotations |
| D-06: Head module included in NB-07 | HONORED | NB-07 cells [8]-[10]: Head module with t vs t_mod distinction |
| D-07: 3 blocks for demo, 30-block config annotated in NB-08 | HONORED | `num_layers=3` in executable cell; cell [6] markdown annotates full 30-block production config |
| D-08: Explicit 48-ch concat from separate tensors in NB-08 | HONORED | `noise_latent`, `control_latent`, `ref_latent` as separate randn before `torch.cat` |
| D-09: Gradient checkpointing train vs eval side-by-side in NB-08 | HONORED | Cell [11]: `model.train()` then checkpoint call; `model.eval()` then no-grad inference; shapes asserted equal |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| NB-06 Exercise 3 (cell [14]) | Wrong LoRA formula for rectangular matrices: uses `2 * max(in, out) * r` instead of `r * (in + out)` | Warning (WR-01) | Educational misinformation — overstates ffn.0 adapter size by 71%; does not affect any assertion or execution |
| NB-06 Summary table (cell [13]) | GateModule.forward cited at line 190 (class definition) not line 194 (forward method) | Info (IN-01) | Minor citation inaccuracy; reader looking up the citation finds the class header, not the forward body |
| NB-08 cells [5] and [14] | WanModel.__init__ citation range `273-328` truncated — body continues through line 338 | Info (IN-02) | Reader checking the cited range misses `has_image_input`, `has_ref_conv`, and `control_adapter` initialization |

No blocker anti-patterns found. No TODOs, FIXMEs, placeholder text, or empty implementations in any notebook.

### Human Verification Required

#### 1. Fix or Accept — WR-01: Wrong LoRA Formula in Exercise 3

**Test:** Open `Course/NB-06-dit-block.ipynb`, scroll to Exercise 3 (final markdown cell). Read the formula: "adapter size = `2 * max(1536, 8960) * r`"

**Expected correct formula:** `r * (in_features + out_features) = 16 * (1536 + 8960) = 167,936`

**Current notebook formula:** `2 * max(1536, 8960) * r = 2 * 8960 * 16 = 286,720` (overstated by 71%)

**Why human:** This is a content quality decision. The incorrect formula does not break execution or any assertion. The code review (WR-01) recommends fixing it. Whether to fix before marking the phase closed, or accept it as an exercise hint that can be corrected later, is a developer call.

**Fix if needed (paste into the Exercise 3 markdown cell):**
```
For ffn.0 (1536->8960): LoRA uses A of shape (r, in_features) and B of shape (out_features, r)
Total = r * (in_features + out_features) = 16 * (1536 + 8960) = 16 * 10,496 = 167,936 parameters
Compression: 13,771,520 / 167,936 = 82x reduction
```

#### 2. Fix or Accept — IN-01: GateModule Citation Line Number

**Test:** Open `Course/NB-06-dit-block.ipynb`, scroll to the Summary cell (second-to-last markdown cell). Find the Source References table row for `GateModule.forward`.

**Expected:** Line 194 (the `def forward` method)

**Current:** Likely shows line 190 (the `class GateModule(nn.Module):` declaration)

**Why human:** Minor accuracy issue; a two-character fix. Needs author sign-off to edit.

#### 3. Fix or Accept — IN-02: WanModel.__init__ Citation Range

**Test:** Open `Course/NB-08-wanmodel-forward.ipynb`. Check cell [5] code comment and the Summary cell's source references table for WanModel.__init__ line range.

**Expected:** Lines 274-338 (includes has_image_input and control_adapter setup)

**Current:** Lines 273-328 or 274-328 (10 lines short)

**Why human:** Minor accuracy issue; needs author decision to update.

---

## Gaps Summary

No functional gaps found. All 8 must-have truths are verified, all 3 notebooks execute without error, all assertions pass, all requirements DIT-12 through DIT-17 are satisfied, and all 9 user decisions from CONTEXT.md are honored.

The three code review findings (WR-01, IN-01, IN-02) are the only outstanding items, all of which are editorial/educational accuracy issues rather than functional blockers. WR-01 (wrong LoRA formula in an exercise) is the most meaningful because it teaches incorrect math; IN-01 and IN-02 are citation accuracy nits.

Status is `human_needed` because WR-01 requires a developer decision on whether to fix or accept before the phase can be declared clean. The notebooks are production-ready for educational use if WR-01 is either fixed or explicitly accepted as tolerable.

---

_Verified: 2026-04-24T16:00:00Z_
_Verifier: Claude (gsd-verifier)_
