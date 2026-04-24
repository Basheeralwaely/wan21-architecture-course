# Phase 4: System Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 04-system-integration
**Areas discussed:** Data flow diagram, Runnable demos

---

## Data Flow Diagram

### Diagram Style

| Option | Description | Selected |
|--------|-------------|----------|
| Single large ASCII diagram | One comprehensive diagram showing full pipeline with tensor shapes at each stage. Consistent with Phase 2/3 ASCII approach. | ✓ |
| Staged diagrams | Break flow into 3 smaller diagrams (encoding, denoising, decoding). Simpler individually but requires mental stitching. | |
| Table-based flow | Stage-by-stage table with columns: Stage, Input, Component, Output, Shape. Compact but less visual. | |

**User's choice:** Single large ASCII diagram
**Notes:** Consistent with established pattern from Phase 2 and 3.

### Notebook Back-References in Diagram

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, with NB refs | Each component box includes notebook number(s) where it was taught. Acts as roadmap. | ✓ |
| No, keep it clean | Just component names and shapes. Concept map handles cross-references separately. | |

**User's choice:** Yes, with NB refs
**Notes:** None

### Denoising Loop Detail

| Option | Description | Selected |
|--------|-------------|----------|
| Walkthrough with scheduler | Show FlowMatchScheduler setup, walk through one step conceptually, explain flow matching velocity prediction. | ✓ |
| Black-box overview | Show loop structure but don't explain flow matching or scheduler internals. | |
| Full loop execution | Run 2-3 steps with dummy tensors through reduced DiT. Most hands-on but heavier. | |

**User's choice:** Walkthrough with scheduler
**Notes:** User specifically requested thorough explanation of denoising — "I don't fully understand it." This is a key teaching priority for NB-12.

---

## Runnable Demos

### Execution Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Component-by-component demos | Create minimal dummy instances of each component separately, run each, show output shapes. Show orchestration conceptually. | ✓ |
| Mini end-to-end pass | Wire all 4 components into a single mini forward pass flowing data through full pipeline. | |
| Conceptual only | No component instantiation. Show pipeline code with annotations only. | |

**User's choice:** Component-by-component demos
**Notes:** None

### Parameter Count Summary

| Option | Description | Selected |
|--------|-------------|----------|
| Live computation | Instantiate components and compute param counts from real code. Consistent with NB-06's LoRA analysis. | ✓ |
| Pre-computed table | Show markdown table with numbers from model_architecture.md. No code needed. | |
| Both | Live computation cells plus summary markdown cell. | |

**User's choice:** Live computation
**Notes:** None

---

## Claude's Discretion

- Import/setup strategy, prose tone, verification design, exercise design
- How much of WanVideoPipeline unit architecture to expose
- Notebook narrative arc (top-down vs bottom-up)
- Whether to reduce dimensions for text encoder and CLIP

## Deferred Ideas

None — discussion stayed within phase scope
