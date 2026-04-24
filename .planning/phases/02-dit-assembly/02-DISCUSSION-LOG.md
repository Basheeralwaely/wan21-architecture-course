# Phase 2: DiT Assembly - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 02-dit-assembly
**Areas discussed:** Composition narrative, LoRA analysis depth, Patchify presentation, Full model execution

---

## Composition Narrative

### Q1: How should NB-06 narrate the DiT Block composition?

| Option | Description | Selected |
|--------|-------------|----------|
| Bottom-up buildup (Recommended) | Start with __init__ showing sub-module wiring, walk through forward() line-by-line with back-references to prior notebooks, end with full forward pass cell | ✓ |
| Top-down decomposition | Show full forward pass first as black box, then zoom into each line | |
| Side-by-side comparison | Show source code forward() alongside annotated cells running each sub-step | |

**User's choice:** Bottom-up buildup
**Notes:** None

### Q2: Should NB-06 include an architecture diagram?

| Option | Description | Selected |
|--------|-------------|----------|
| ASCII diagram (Recommended) | Simple ASCII/text block diagram showing data flow between sub-modules with residual connections | ✓ |
| No diagram | Let step-by-step code walkthrough speak for itself, consistent with NB-01 through NB-05 | |
| Mermaid/rendered diagram | Use mermaid diagram cell that renders in Jupyter — adds dependency | |

**User's choice:** ASCII diagram
**Notes:** None

---

## LoRA Analysis Depth

### Q1: How deep should the LoRA target module analysis go?

| Option | Description | Selected |
|--------|-------------|----------|
| Counts + rationale (Recommended) | Parameter counts per sub-module with percentages, explain WHY these layers are LoRA targets, no LoRA math | ✓ |
| Counts only | Just parameter count table and target module identification, no explanation | |
| Full LoRA explainer | Counts + rationale + LoRA rank explanation, A×B decomposition, rank-capacity relationship | |

**User's choice:** Counts + rationale
**Notes:** None

### Q2: Should parameter count be per block, full model, or both?

| Option | Description | Selected |
|--------|-------------|----------|
| Both (Recommended) | Per-block breakdown first (detailed), then multiply to full 30-block model total | ✓ |
| Per block only | One block's anatomy, 30x multiplication left as exercise | |
| Full model only | Aggregate counts across all 30 blocks | |

**User's choice:** Both
**Notes:** None

---

## Patchify Presentation

### Q1: Should NB-07 include ASCII diagrams for spatial-to-sequence mapping?

| Option | Description | Selected |
|--------|-------------|----------|
| ASCII diagrams (Recommended) | Visual showing grid carved into patches and flattened into sequence, consistent with NB-06 diagram decision | ✓ |
| Code and shapes only | Code cells with shape annotations, minimal and consistent with NB-01–NB-05 | |
| Both code + diagram | Lead with ASCII diagram, follow with code walkthrough | |

**User's choice:** ASCII diagrams
**Notes:** None

### Q2: Should NB-07 also cover the Head module?

| Option | Description | Selected |
|--------|-------------|----------|
| Include in NB-07 (Recommended) | Head module pairs naturally with patchify/unpatchify as output-side counterpart, keeps NB-08 focused | ✓ |
| Save for NB-08 | Head only makes sense in full model forward pass context | |

**User's choice:** Include in NB-07
**Notes:** None

---

## Full Model Execution

### Q1: How should NB-08 handle the 30-block WanModel for CPU execution?

| Option | Description | Selected |
|--------|-------------|----------|
| Reduced blocks (Recommended) | Instantiate with 2-3 blocks for runnable cells, show real 30-block config in annotation cell | ✓ |
| Full 30 blocks | Run real 30-block model on CPU, may break STD-03 5-second limit | |
| Both configurations | Reduced blocks first, then optional slow cell with full 30 blocks | |

**User's choice:** Reduced blocks
**Notes:** None

### Q2: How should NB-08 present the 48-channel input composition?

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit concat demo (Recommended) | Create separate dummy tensors for noise/control/reference, show torch.cat, feed into model | ✓ |
| Single 48-ch tensor + prose | Create 48-channel dummy directly, explain composition in prose | |

**User's choice:** Explicit concat demo
**Notes:** None

### Q3: How should NB-08 handle gradient checkpointing (DIT-17)?

| Option | Description | Selected |
|--------|-------------|----------|
| Show stripping for inference (Recommended) | Training vs inference code paths side-by-side, explain memory savings vs recomputation | ✓ |
| Brief mention only | One paragraph noting it exists and is disabled during inference | |

**User's choice:** Show stripping for inference
**Notes:** None

---

## Claude's Discretion

- Import/setup strategy (same pattern as Phase 1)
- Exact prose length and tone
- Verification cell design beyond shape assertions
- Exercise design (2-3 per notebook)
- Whether to reduce dim alongside block count in NB-08

## Deferred Ideas

None — discussion stayed within phase scope
