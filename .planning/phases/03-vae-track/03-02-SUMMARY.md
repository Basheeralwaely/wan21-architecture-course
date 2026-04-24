---
phase: 03-vae-track
plan: "02"
subsystem: Course/notebooks
tags: [vae, encoder, decoder, resample, reparameterization, patchify, jupyter]
dependency_graph:
  requires: [NB-09-causalconv3d-resblock-attn]
  provides: [NB-10-vae-encoder, NB-11-vae-decoder-patchify]
  affects: [Phase-4-full-pipeline]
tech_stack:
  added: []
  patterns: [importlib-path-search, tqdm-stub, reduced-dim-demo, cpu-execution, empirical-shape-verification]
key_files:
  created:
    - Course/NB-10-vae-encoder.ipynb
    - Course/NB-11-vae-decoder-patchify.ipynb
  modified: []
decisions:
  - "Auto-fixed: Encoder3d z_dim must be 8 (not 4) for mu/log_var split -- plan said z_dim=4 gives [1,8,...] output but actual source gives [1,4,...]; VideoVAE_ passes z_dim*2 to Encoder3d"
  - "Auto-fixed: upsample2d also halves channels -- plan said channels unchanged for upsample2d but actual Conv2d(dim, dim//2) halves for both upsample2d and upsample3d"
  - "Auto-fixed: decoder dims are [32,32,32,16,8] not [32,32,16,8,8] -- verified empirically from actual Decoder3d source (dim=8, dim_mult=[1,2,4,4])"
  - "Auto-fixed: time_conv demo input must be T=5 to show temporal halving (T=5: (5-3)//2+1=2) -- T=4 gives T_out=1 which is not T//2"
metrics:
  duration: "~18m"
  completed: "2026-04-24"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 3 Plan 02: NB-10 VAE Encoder + NB-11 VAE Decoder/Patchify Summary

## One-Liner

VAE encoder 4-level downsampling with Resample internals and reparameterization (NB-10), decoder mirror-plus-differences with in_dim halving and VAE vs DiT patchify disambiguation (NB-11), both importlib-loaded from `wan_video_vae.py`.

## What Was Built

### NB-10: `Course/NB-10-vae-encoder.ipynb` (13 cells)

A Jupyter notebook tracing the full `Encoder3d` downsampling pathway following the Phase 1/2 template (STD-01 through STD-07).

**Cell map:**
- Cell 0 (md): Title, learning objectives (3), prerequisites (NB-09), concept map with ASCII data-flow diagram
- Cell 1 (code): Setup -- tqdm stub, 10-level path search for `wan_video_vae.py`, importlib load, all VAE imports
- Cell 2 (md): Encoder3d structure -- sub-module overview table (conv1, downsamples, middle, head)
- Cell 3 (code): Reduced-scale encoder instantiation `Encoder3d(dim=8, z_dim=8, ...)` with downsamples breakdown
- Cell 4 (md): Resample module internals prose (D-05) -- downsample2d vs downsample3d, feat_cache acknowledgment
- Cell 5 (code): Resample internals demo -- both modes, shape assertions
- Cell 6 (code): time_conv direct demo -- temporal stride shown on T=5 input -> T=2 output
- Cell 7 (md): Level-by-level shape trace tables -- demo (dim=8) and production (dim=96) annotation
- Cell 8 (code): Full encoder forward pass with shape assertions
- Cell 9 (md): Reparameterization prose (D-06) -- mu/log_var split explanation
- Cell 10 (code): Reparameterization demo -- chunk(2,dim=1), eps*std+mu formula, assertions
- Cell 11 (md): Summary with source references table
- Cell 12 (md): Exercises (3 exercises)

### NB-11: `Course/NB-11-vae-decoder-patchify.ipynb` (17 cells)

A Jupyter notebook covering `Decoder3d` upsampling (mirror of encoder) and VAE vs DiT patchify disambiguation.

**Cell map:**
- Cell 0 (md): Title, learning objectives (3), prerequisites (NB-09, NB-10, NB-07), concept map with ASCII decoder data-flow
- Cell 1 (code): Setup -- same as NB-10
- Cell 2 (md): Decoder3d mirror-plus-differences framing (D-07) -- comparison table with 4 sub-modules
- Cell 3 (code): Reduced-scale decoder instantiation `Decoder3d(dim=8, z_dim=4, ...)` with upsamples breakdown
- Cell 4 (md): in_dim halving prose -- Pitfall 2 explanation
- Cell 5 (code): in_dim halving walkthrough -- shows level 0-3 with original vs used in_dim
- Cell 6 (md): Decoder shape trace table -- demo shapes (dim=8, input [1,4,5,2,2])
- Cell 7 (code): Encode-decode round-trip with shape assertions
- Cell 8 (md): Upsample Resample internals prose
- Cell 9 (code): Upsample Resample internals -- both modes show channel halving, assertions
- Cell 10 (md): VAE vs DiT patchify comparison table (D-08, VAE-05)
- Cell 11 (md): ASCII patchify disambiguation diagram
- Cell 12 (code): VAE patchify demo -- `patchify(x_5d, patch_size=2)` produces `[1,64,4,4,4]`
- Cell 13 (code): Round-trip: `torch.allclose(unpatchify(patchify(x)), x)` returns True
- Cell 14 (code): Parameter comparison -- 0 params (VAE) vs 296,448 params (DiT)
- Cell 15 (md): Summary with source references table
- Cell 16 (md): Exercises (3 exercises)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed encoder z_dim: use z_dim=8 (not 4) for mu/log_var split**
- **Found during:** Task 1 -- empirical verification before writing notebook
- **Issue:** Plan specified `Encoder3d(dim=8, z_dim=4, ...)` and claimed it outputs `[1, 8, 5, 2, 2]` (z_dim*2=8 channels). The actual `Encoder3d.__init__` (line 567) sets head output to `z_dim` channels (not `z_dim*2`). With `z_dim=4`, the encoder outputs `[1, 4, 5, 2, 2]`. `VideoVAE_` passes `z_dim*2` to Encoder3d (line 971), so to get 8-channel output for `chunk(2, dim=1)`, the encoder must be instantiated with `z_dim=8`.
- **Fix:** Used `Encoder3d(dim=8, z_dim=8, ...)` throughout NB-10. Updated all shape assertions and comments to match actual output `[1, 8, 5, 2, 2]`.
- **Files modified:** `Course/NB-10-vae-encoder.ipynb`
- **Commit:** 426963d

**2. [Rule 1 - Bug] Fixed upsample2d channel behavior: halves channels (same as upsample3d)**
- **Found during:** Task 2 -- empirical verification before writing NB-11
- **Issue:** Plan said "channels UNCHANGED for upsample2d" and claimed `Resample(8, 'upsample2d')` outputs 8 channels. The actual source (lines 92-95) uses `Conv2d(dim, dim // 2, 3, padding=1)` for BOTH `upsample2d` and `upsample3d`. Both modes halve channels. Empirically verified: `Resample(8, 'upsample2d')([1,8,4,4,4]) -> [1,4,4,8,8]`.
- **Fix:** Updated NB-11 cells 9 and 6 (shape table) to show correct channel halving for upsample2d. Updated the decoder shape trace accordingly. Added note in the comparison section.
- **Files modified:** `Course/NB-11-vae-decoder-patchify.ipynb`
- **Commit:** fbcefe6

**3. [Rule 1 - Bug] Fixed decoder dims: [32,32,32,16,8] not [32,32,16,8,8]**
- **Found during:** Task 2 -- empirical verification of decoder dims computation
- **Issue:** Plan's in_dim halving walkthrough showed `dims = [8, 16, 32, 32, 32]` reversed as `[32, 32, 16, 8, 8]`. The actual source line 755: `dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]`. With `dim=8, dim_mult=[1,2,4,4]`: `[dim_mult[-1]] + dim_mult[::-1] = [4,4,2,1]`, giving `dims = [32, 32, 16, 8]`... actually `[dim_mult[-1]] = [4]` and `dim_mult[::-1] = [4,4,2,1]`, so `[4,4,4,2,1]`, giving `dims = [32, 32, 32, 16, 8]`. Verified empirically.
- **Fix:** Updated NB-11 cell 5 with correct dims `[32, 32, 32, 16, 8]` and correct level descriptions.
- **Files modified:** `Course/NB-11-vae-decoder-patchify.ipynb`
- **Commit:** fbcefe6

**4. [Rule 1 - Bug] Fixed time_conv demo input: must use T=5 (not T=4) to show temporal halving**
- **Found during:** Task 1 -- empirical test of `time_conv` with T=4
- **Issue:** Plan used `x = torch.randn(1, 8, 4, 8, 8)` for the time_conv demo, claiming it would output T=2 (temporal halved). With T=4 and kernel_t=3, stride_t=2, no padding: output_T = (4-3)//2+1 = 1 (not 2). T=4 gives T_out=1.
- **Fix:** Used `x = torch.randn(1, 8, 5, 8, 8)` (T=5). Formula: (5-3)//2+1 = 2. Shows temporal halving correctly. Added inline comment explaining the formula.
- **Files modified:** `Course/NB-10-vae-encoder.ipynb`
- **Commit:** 426963d

## Verification Results

**NB-10 structural checks (17/17 PASS):**
```
PASS: cell_count (13 cells)
PASS: has_Encoder3d
PASS: has_Resample
PASS: has_dim_mult
PASS: has_shape_table
PASS: has_time_conv
PASS: has_reparameterize
PASS: has_mu_logvar
PASS: has_chunk_split
PASS: has_reduced_scale
PASS: has_shape_assert
PASS: has_exercises
PASS: has_summary
PASS: has_source_citation
PASS: has_prerequisites
PASS: has_importlib
PASS: has_feat_cache_note
Overall: PASS (17/17)
jupyter nbconvert --execute: SUCCESS
```

**NB-11 structural checks (18/18 PASS):**
```
PASS: cell_count (17 cells)
PASS: has_Decoder3d
PASS: has_patchify
PASS: has_unpatchify
PASS: has_mirror_framing
PASS: has_in_dim_halving
PASS: has_comparison_table
PASS: has_round_trip
PASS: has_upsample
PASS: has_reduced_scale
PASS: has_shape_assert
PASS: has_exercises
PASS: has_summary
PASS: has_source_citation
PASS: has_prerequisites
PASS: has_importlib
PASS: has_ascii_diagram
PASS: has_nb07_backref
Overall: PASS (18/18)
jupyter nbconvert --execute: SUCCESS
```

## Known Stubs

None. All code cells use real `diffsynth.models.wan_video_vae` classes imported via importlib. No hardcoded data, no mock outputs.

## Threat Flags

None. Notebooks are read-only educational files with dummy tensors and public architecture walkthrough code.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Create NB-10 | 426963d | feat(03-02): create NB-10 VAE encoder notebook |
| Task 2: Create NB-11 | fbcefe6 | feat(03-02): create NB-11 VAE decoder + patchify disambiguation notebook |

## Self-Check: PASSED

- `Course/NB-10-vae-encoder.ipynb`: FOUND
- `Course/NB-11-vae-decoder-patchify.ipynb`: FOUND
- Commit `426963d`: FOUND (verified via git log)
- Commit `fbcefe6`: FOUND (verified via git log)
- NB-10 executes via `jupyter nbconvert --execute`: PASSED (all assertions pass)
- NB-11 executes via `jupyter nbconvert --execute`: PASSED (all assertions pass)
- NB-10 structural checks 17/17: PASSED
- NB-11 structural checks 18/18: PASSED
- No stubs found in either notebook
- Full encode+decode timing: 24ms (well under 5s STD-03 limit)
