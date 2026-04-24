# Wan2.1 Model Architecture Course

## What This Is

A series of Jupyter notebook tutorials in `Course/` that teach the internal architecture of the Wan2.1 motion control video diffusion model. The notebooks walk through the actual DiffSynth source code bottom-up — from individual attention layers and QKV projections, through DiT blocks, up to the full DiT model and VAE encoder/decoder. Targeted at intermediate PyTorch practitioners who need diffusion and DiT concepts explained.

## Core Value

Clear, annotated code walkthroughs that make the Wan2.1 architecture understandable — each notebook builds on the last so readers can trace how basic components compose into the full model.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Bottom-up notebook sequence covering attention, QKV, DiT blocks, full DiT, VAE encoder/decoder
- [ ] Code walkthrough style — annotate actual source code from `diffsynth/models/`
- [ ] Lightweight runnable cells with dummy tensors to show shapes and data flow
- [ ] Intermediate audience — explain diffusion/DiT concepts, assume PyTorch basics
- [ ] No modifications to existing codebase files
- [ ] All notebooks live in `Course/` directory

### Out of Scope

- Training tutorials — this covers architecture only, not how to train
- Inference walkthroughs — covered separately in existing scripts
- Beginner PyTorch fundamentals — audience is expected to know tensors, modules, forward passes
- Full model weight loading — notebooks use lightweight dummy tensors instead

## Context

- Existing codebase is a Wan2.1-Fun-V1.1-1.3B-Control LoRA training pipeline built on DiffSynth
- Key model files: `diffsynth/models/wan_video_dit.py` (DiT with 30 blocks, ~780M params), `diffsynth/models/wan_video_vae.py` (video autoencoder), `diffsynth/models/wan_video_text_encoder.py` (T5/UMT5), `diffsynth/models/wan_video_image_encoder.py` (CLIP)
- The codebase has already been mapped — architecture docs in `.planning/codebase/`
- Model uses flow matching diffusion with multi-modal conditioning (text, control video, reference image, CLIP embeddings)

## Constraints

- **Read-only codebase**: Existing files must not be modified — only new notebooks added
- **Directory**: All new content goes in `Course/`
- **Lightweight execution**: Notebooks should run without downloading full model weights (use dummy tensors for shape demonstrations)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bottom-up notebook ordering | Build understanding from primitives to composed systems | -- Pending |
| Dummy tensors over real weights | Keep notebooks fast and portable without GPU/model requirements | -- Pending |
| Code walkthrough over theory-first | Audience learns better by reading annotated real code | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-24 after initialization*
