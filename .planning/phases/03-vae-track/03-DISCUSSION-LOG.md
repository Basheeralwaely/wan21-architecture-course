# Phase 3: VAE Track - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 03-vae-track
**Areas discussed:** CausalConv3d teaching, Encoder depth, Patchify disambiguation

---

## CausalConv3d Teaching

| Option | Description | Selected |
|--------|-------------|----------|
| Visual-first (Recommended) | ASCII diagram showing padding prevents future frame leakage, then derive formula from diagram | ✓ |
| Code-walkthrough | Walk through __init__ padding computation and forward() F.pad application | |
| Theory-then-code | Start with why causal structure matters, then derive padding, then show code | |

**User's choice:** Visual-first
**Notes:** Consistent with Phase 2's ASCII diagram approach (D-02, D-05)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Skip connection focus (Recommended) | Emphasize identity vs learned 1x1 projection skip path | |
| Full conv pipeline | Walk entire residual sequential layer by layer showing tensor transforms | ✓ |
| You decide | Claude picks | |

**User's choice:** Full conv pipeline
**Notes:** User wants thorough layer-by-layer walkthrough of ResBlock internals

---

| Option | Description | Selected |
|--------|-------------|----------|
| Include in NB-09 (Recommended) | Keep all VAE primitives in one notebook | ✓ |
| Defer to NB-10 | AttentionBlock appears inline in encoder walkthrough | |

**User's choice:** Include in NB-09
**Notes:** Keeps primitives-first pattern consistent with Phase 1

---

## Encoder Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full level-by-level (Recommended) | Walk all 4 levels with shape table at each stage | ✓ |
| Pattern explanation | Show level 0 in detail, explain pattern repeats, summary table | |
| You decide | Claude picks | |

**User's choice:** Full level-by-level
**Notes:** Shape table from Input through all levels to Latent output

---

| Option | Description | Selected |
|--------|-------------|----------|
| Code-focused (Recommended) | Show mu/log_var split, reparameterize(), scale normalization. Brief WHY without ELBO theory. | ✓ |
| Theory + code | Variational inference prose, KL divergence motivation, then code | |
| Minimal | Just code and shapes, one-liner note | |

**User's choice:** Code-focused
**Notes:** Brief explanation of why reparameterization enables backprop, no deep VAE theory

---

| Option | Description | Selected |
|--------|-------------|----------|
| Show internals (Recommended) | Walk spatial path (ZeroPad2d + strided Conv2d) and temporal path (CausalConv3d stride) | ✓ |
| Black box | Treat as 'downsamples dimensions', reference NB-09 | |
| You decide | Claude picks | |

**User's choice:** Show internals
**Notes:** New operations not covered in NB-09

---

## Patchify Disambiguation

| Option | Description | Selected |
|--------|-------------|----------|
| Side-by-side section (Recommended) | Dedicated section with comparison table and ASCII diagrams. Back-reference NB-07. | ✓ |
| Inline callout | Callout box when VAE patchify first appears in decoder walkthrough | |
| Separate comparison cell | Run both on same dummy tensor, print shapes | |

**User's choice:** Side-by-side section
**Notes:** VAE patchify = einops channel rearrangement (no params) vs DiT patchify = Conv3d learned projection

---

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror + differences (Recommended) | Start with "mirrors encoder" framing, reverse shape table, focus on differences | ✓ |
| Full independent walkthrough | Walk decoder from scratch as if encoder didn't exist | |
| You decide | Claude picks | |

**User's choice:** Mirror + differences
**Notes:** Focus on what's different: upsample Resample, extra ResBlock per level, reversed dim_mult, channel halving

---

## Claude's Discretion

- Import/setup strategy (established in Phase 1)
- Prose length and tone
- Verification cell design
- Exercise design (2-3 modification exercises per notebook)
- feat_cache handling in code examples (user did not select this for discussion)
- Reduced encoder/decoder depth for runnable cells

## Deferred Ideas

None — discussion stayed within phase scope
