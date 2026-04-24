# Feature Landscape

**Domain:** ML model architecture tutorial notebooks (DiT / VAE / video diffusion)
**Project:** Wan2.1 Model Architecture Course
**Researched:** 2026-04-24

---

## Context

This project produces a bottom-up sequence of Jupyter notebooks in `Course/` explaining
the internal architecture of the Wan2.1-Fun-V1.1-1.3B-Control model. The source material
is the actual `diffsynth/` codebase — notebooks annotate real production code, not toy
rewrites. Audience: intermediate PyTorch practitioners who know tensors and `nn.Module`
but need diffusion/DiT/VAE concepts explained.

The features below are evaluated against this specific context, not against general ML
tutorial notebooks at large.

---

## Table Stakes

Features readers expect. Missing = notebook feels broken or incomplete, reader leaves.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Tensor shape annotations on every operation | Readers need to track how `[B,S,D]` transforms step-by-step — this is the primary debugging/understanding tool for architecture readers | Low | Do this as inline comments `# [B, 21504, 1536]` immediately after each reshape, projection, or attention call. Annotate actual values (not symbols) at the working resolution (512×512, 81 frames) |
| Runnable cells with dummy tensors | Readers must be able to execute and see real output without downloading 780M-param weights | Low | Use `torch.randn` / `torch.zeros` shaped to match real model dims. Cells should complete on CPU in under 5s |
| `print(tensor.shape)` confirmation cells | Every non-obvious shape transformation needs a verification cell — standard "sanity check" expectation in this genre | Low | Not optional; readers copy these into their own debugging sessions |
| Prose explanation before each code block | Code-only cells are unreadable for architectural concepts; every block needs 2–5 sentences of "what this layer does and why" | Low | Order: explanation → code → shape check. Never code first |
| Notebook ordering that builds bottom-up | Readers expect to be able to read linearly: attention primitives before DiT blocks, DiT blocks before full model | Low | Dependency: 01_attention → 02_QKV → 03_DiT_block → 04_full_DiT → 05_VAE. Later notebooks can import helpers from earlier ones but must re-explain concepts briefly |
| Source code references with line numbers | Readers will cross-reference actual files; every class/function must cite `diffsynth/models/wan_video_dit.py:L125` | Low | Makes the notebook a reading companion, not a rewrite |
| Section headers mapping to the architecture diagram | Readers orient by the architecture diagram in `model_architecture.md`; notebook headers must match diagram box names | Low | E.g. "SelfAttention", "CrossAttention", "DiT Block", "Time Embedding", "Patch Embedding", "VAE Encoder" |
| Clear prerequisite statement per notebook | Readers must know what to read first; what PyTorch concepts are assumed | Low | One cell at the top: "Prerequisites: understand `nn.Module.forward()`, `torch.einsum`, `rearrange`" |

---

## Differentiators

Features that distinguish a high-quality walkthrough from a generic tutorial. Not
universally expected, but what separates good from forgettable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Annotate real production code, not a reimplementation | Readers studying Wan2.1 specifically want to understand the code they will actually train or fork — walkthroughs of toy reimplementations don't transfer | Medium | Import from `diffsynth` directly (e.g., `from diffsynth.models.wan_video_dit import SelfAttention`) and instantiate real classes with dummy inputs. Show actual parameter counts |
| Side-by-side diagram cells using ASCII or matplotlib | DiT block data flow and attention shape transforms are hard to grasp from prose alone; a per-notebook diagram anchors everything | Medium | ASCII diagrams inline in markdown cells — low friction, no image files needed. The `model_architecture.md` diagrams are good references to adapt |
| "Why not X?" design decision callouts | The Wan2.1 DiT makes specific choices (RMSNorm not LayerNorm on QK, RoPE not sinusoidal, CrossAttention separate from SelfAttention, gated FFN, adaLN-Zero modulation) — explaining why each choice was made versus alternatives is high signal for the intended audience | Medium | Callout cells marked `> [Design note]`. Cover: RMSNorm vs LayerNorm, 3D RoPE vs 2D, adaLN-Zero vs in-projection conditioning, flow matching vs DDPM noise prediction |
| Concrete parameter count breakdowns | Intermediate practitioners learning LoRA targeting need to see exactly which layers have how many parameters, why Q/K/V/O are LoRA targets | Low | Cell: `sum(p.numel() for p in layer.parameters())` on real instantiated modules. Connect to LoRA rank-32 decision: 4×(1536×32+32×1536) per block |
| "What does bad input look like?" cells | Showing what happens when tensor shapes are wrong (and the error message) builds debugging intuition that textbooks skip | Low | Add one deliberate shape-mismatch cell per notebook with the expected RuntimeError, followed by the fix |
| 3D RoPE walkthrough as its own focused section | RoPE in 3D (temporal + height + width separate frequencies) is novel and specific to video models; no general tutorial covers it | High | Cover: `precompute_freqs_cis_3d`, `rope_apply`, what `torch.view_as_complex` is doing, why complex multiplication encodes relative position. This is a genuine gap in public tutorials |
| FlashAttention backend comparison cell | The code switches between FlashAttn 3, FlashAttn 2, SageAttention, and `F.scaled_dot_product_attention` based on what's installed — intermediate readers want to understand why and what changes | Low | Show the fallback chain from `flash_attention()` in `wan_video_dit.py:L28-61`; benchmarking one call with `%timeit` is achievable without GPU |
| VAE temporal causality explanation | `CausalConv3d` with asymmetric padding is not intuitive — this is specific to video VAEs and has no analogue in image VAEs | Medium | Show the `_padding` asymmetry logic, why causal structure prevents temporal leakage, connect to the 4× temporal compression (81 frames → 21 latents) |
| Explicit "what gets passed between notebooks" summary | Each notebook should end with a cell summarizing what tensors it produced and what the next notebook expects | Low | Creates a mental ledger. Example: "Notebook 01 produces `q, k, v` shaped `[B, heads, S, head_dim]`; Notebook 02 uses these to demonstrate multi-head composition" |

---

## Anti-Features

Things to deliberately NOT build in these notebooks.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Loading real model weights | Requires ~8GB VRAM, HuggingFace authentication, network download — breaks portability and the "run in 5 seconds" promise | Use dummy tensors shaped to match real dims. Where exact weight values matter (e.g., showing trained embeddings), load from `cache/*.pt` sample files which are already small and local |
| Training loops | Out of scope per PROJECT.md; adds complexity that distracts from architecture explanation | One sentence explaining "this model is trained with flow matching loss — see `train_lora_cached.py`" with a pointer, then move on |
| Inference pipeline walkthroughs | Already covered in existing scripts; this course is architecture only | Reference `inference.py` for the scheduler loop; don't duplicate it |
| Modifying `diffsynth/` source files | Hard constraint in PROJECT.md — read-only codebase | All code in notebook cells; import from `diffsynth` without touching its files |
| Beginner PyTorch fundamentals recap | Audience knows PyTorch basics; covering `nn.Linear` from scratch wastes their time and signals wrong audience | State prerequisites clearly and link to external resources (PyTorch docs) for those who need them |
| GPU-required cells | Breaks portability; intermediate readers often work on CPU laptops for study | All cells run on CPU. Mark GPU-only operations explicitly with `# Note: this cell runs slowly on CPU — expected` |
| Comprehensive test coverage / pytest | This is a tutorial, not a library — formal tests add friction without reader value | Use assertion cells (`assert tensor.shape == (1, 12, 22528, 128)`) to confirm expected outputs; these serve the same didactic purpose |
| Interactive widgets (ipywidgets) | Adds dependency weight; fragile in different Jupyter environments | Static print output and matplotlib figures are sufficient and more portable |
| Separate explanation blog posts or docs | The notebook IS the document — splitting explanation out of the notebook breaks the code-prose coupling that makes this format valuable | Keep prose in markdown cells directly above the code they explain |
| Over-abstracting shared helpers into a `utils.py` | Readers need every step visible; hidden helper functions require jumping to another file, breaking the linear reading experience | Copy-paste the few lines needed in each notebook; accept some redundancy |

---

## Feature Dependencies

```
Tensor shape annotations
  └─ required by: every other feature (shapes appear in all cells)

Runnable dummy tensor cells
  └─ enables: shape annotation verification, "bad input" cells, parameter count cells
  └─ required by: 3D RoPE walkthrough, FlashAttention backend comparison

Source code references
  └─ required by: annotating real production code

Bottom-up notebook ordering
  └─ enables: each notebook assumes prior notebooks are read
  └─ sequence: 01_attention_primitives → 02_qkv_projections → 03_dit_block → 04_full_dit → 05_vae

3D RoPE walkthrough (Differentiator)
  └─ depends on: attention primitives notebook (01), QKV projections notebook (02)

VAE temporal causality explanation (Differentiator)
  └─ independent of DiT notebooks — can be notebook 05 without depending on 01-04

adaLN-Zero design decision callout (Differentiator)
  └─ depends on: DiT block notebook (03)

Parameter count breakdowns (Differentiator)
  └─ depends on: runnable dummy tensor cells
  └─ most useful in: full DiT notebook (04) and DiT block notebook (03)

"What does bad input look like?" cells (Differentiator)
  └─ depends on: runnable dummy tensor cells
  └─ one per notebook, low effort, high payoff
```

---

## MVP Recommendation

For an initial deliverable covering the core learning loop, prioritize:

**Must ship in MVP (table stakes + highest-signal differentiators):**
1. All table stakes features — non-negotiable
2. Annotate real production code (not toy reimplementation) — this is the core value
3. Side-by-side ASCII diagram cells — easy, high impact
4. Parameter count breakdowns — directly supports LoRA understanding
5. "Design decision" callouts for RMSNorm, adaLN-Zero, and 3D RoPE — unique content that can't be found in general DiT tutorials

**Defer from MVP:**
- 3D RoPE deep-dive notebook (high complexity, medium audience need) — add as notebook 06 after core five are shipped
- "Bad input" error cells — low effort but low priority; add in revision pass
- FlashAttention backend comparison — informational but not architectural; add as an appendix section

---

## Notebook Scope Map

Based on feature analysis and architecture depth:

| Notebook | Core Content | Table Stakes to Cover | Key Differentiators |
|----------|--------------|-----------------------|---------------------|
| `01_attention_primitives.ipynb` | `flash_attention()`, `RMSNorm`, `AttentionModule`, scaled dot-product attention | Shape annotations, dummy tensors, source refs | FlashAttention backend chain, RMSNorm vs LayerNorm design note |
| `02_qkv_and_rope.ipynb` | `SelfAttention`, `CrossAttention`, QKV projections, `rope_apply`, `precompute_freqs_cis_3d` | Shape annotations, dummy tensors, source refs | 3D RoPE walkthrough, "relative position via complex multiplication" explanation |
| `03_dit_block.ipynb` | `DiTBlock`, `modulate()`, `GateModule`, adaLN-Zero modulation, time embedding flow | Shape annotations, dummy tensors, source refs | adaLN-Zero design callout, gate_msa / gate_mlp intuition, parameter count per block |
| `04_full_dit.ipynb` | `WanModel` end-to-end: patch embed, ref_conv, time projection, 30 blocks, output head, unpatchify | Shape annotations, dummy tensors, source refs | Full param count, input tensor concatenation (noise+control+ref=[B,48,...]), LoRA target identification |
| `05_vae.ipynb` | `CausalConv3d`, `ResBlock`, encoder downsampling levels, reparameterization, decoder upsample levels | Shape annotations, dummy tensors, source refs | Temporal causality explanation, 4× temporal / 8× spatial compression derivation |

---

## Sources

- Codebase: `diffsynth/models/wan_video_dit.py` — actual `SelfAttention`, `CrossAttention`, `DiTBlock`, `WanModel` implementations
- Codebase: `diffsynth/models/wan_video_vae.py` — `CausalConv3d`, `RMS_norm`, VAE level structure
- Project context: `.planning/PROJECT.md` — audience definition, constraints, out-of-scope
- Architecture reference: `model_architecture.md` — tensor shape diagrams used throughout
- Prior art (MEDIUM confidence): [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) — gold standard for "annotated real code" format
- Prior art (MEDIUM confidence): [labmlai annotated paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — side-by-side code+notes format
- Prior art (MEDIUM confidence): [Karpathy "Let's build GPT"](https://github.com/karpathy/nn-zero-to-hero) — build-from-primitives ordering rationale
- DiT reference (HIGH confidence): [Facebook Research DiT](https://github.com/facebookresearch/DiT) — adaLN-Zero is the empirically best conditioning approach
- RoPE reference (MEDIUM confidence): [EleutherAI RoPE blog](https://blog.eleuther.ai/rotary-embeddings/) — relative position encoding via complex rotation
- adaLN-Zero (HIGH confidence): [DiT paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) — adaLN-Zero best FID with minimal Gflops overhead
