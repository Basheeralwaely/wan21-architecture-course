# Domain Pitfalls: ML Architecture Tutorial Notebooks

**Domain:** Educational code walkthrough notebook series — DiT/VAE/attention architecture internals
**Project:** Wan2.1 Model Architecture Course
**Researched:** 2026-04-24

---

## Critical Pitfalls

Mistakes that cause readers to misunderstand the architecture, break cells silently, or abandon the notebook.

---

### Pitfall 1: Dummy Tensors With Wrong Shapes

**What goes wrong:** A cell creates a dummy input like `torch.randn(1, 16, 8, 8)` to demonstrate a module, but that shape is incorrect for the actual Wan2.1 forward pass. The cell still runs, the output tensor exists, but the shape story told to the reader is false. Later notebooks build on wrong mental models.

**Why it happens:** Authors create shapes intuitively ("batch of 1, 16 channels, small spatial") rather than deriving them from actual model config. The Wan2.1 DiT expects latent input `(B, C, F, H, W)` where channels=16 (`in_dim`), frames/height/width must be divisible by the 3D patch size `(1, 2, 2)`. A dummy shape of `(1, 16, 4, 32, 32)` passes silently, but `(1, 16, 1, 9, 9)` crashes unpatchify because 9 is not divisible by 2.

**Consequences:** Reader either sees an unexplained error, or sees a valid output with misleading dimensions. Shape intuition developed across notebooks will not transfer to reading real training code.

**Prevention:**
- Derive all dummy tensor shapes from the actual model config constants at the top of each notebook (e.g., `DIM = 1536`, `NUM_HEADS = 12`, `HEAD_DIM = DIM // NUM_HEADS`, `PATCH_SIZE = (1, 2, 2)`).
- Run the full forward pass with actual module instances using dummy tensors before writing prose, then copy the verified shapes into cell comments.
- Add explicit `assert output.shape == expected` cells after every shape demonstration.

**Warning signs:** A dummy input that doesn't reference any config constants; a cell whose prose says "shape is (B, T, D)" but the code says something different; a notebook that works but produces a different shape than `wan_video_dit.py` would produce at inference.

**Phase:** Every notebook phase. Address in Phase 1 (attention/QKV) by establishing the shape convention reference cells that all subsequent notebooks import.

---

### Pitfall 2: Flattening the Attention Layout Explanation

**What goes wrong:** The tutorial explains QKV as "Q, K, V are each shape (B, S, D)" and moves on. This hides the critical reshape: QKV exist in that flat form only for a moment — the actual computation requires head-split form, and the Wan2.1 code uses two different head layouts depending on which attention backend is active:
- Flash Attention 2/3: `(B, S, N, D_head)` — sequence-first
- Compatibility mode / SageAttn: `(B, N, S, D_head)` — heads-first

A reader who doesn't see both conventions in the same notebook will be confused the moment they look at the real `flash_attention()` function.

**Why it happens:** Most textbook attention tutorials pick one layout and never mention the other exists. The Wan2.1 source switches layout inside the same function based on runtime availability flags.

**Consequences:** Reader cannot trace what `rearrange(q, "b s (n d) -> b n s d", n=num_heads)` vs `rearrange(q, "b s (n d) -> b s n d", n=num_heads)` means, or why both patterns appear in the same file.

**Prevention:**
- Dedicate one notebook cell to showing both layout conventions side-by-side with explicit shape printouts.
- Show the `flash_attention()` function's conditional layout selection as a deliberate design decision, not an implementation detail to skip.
- Annotate every `rearrange` call with the before/after shape in a comment.

**Warning signs:** A notebook cell that calls `rearrange` without a comment showing the concrete shape transformation; prose that calls it "just a reshape" without explaining the head-sequence order choice.

**Phase:** Notebook on `SelfAttention` / `AttentionModule`. Must address before any subsequent notebook that touches cross-attention.

---

### Pitfall 3: Treating adaLN Modulation as "Just Conditioning"

**What goes wrong:** The tutorial says "the timestep conditions the block via adaptive layer norm" and shows the six-chunk split as boilerplate. The reader never understands *why* there are six parameters, what each one does, or why the gate parameters start near zero (adaLN-Zero initialization in the original DiT paper).

**Why it happens:** The six-chunk pattern looks mechanical. `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp` is a wall of names. Authors tend to just show the code and explain what the names mean without explaining why the split is six and not two, four, or eight.

**Consequences:** Reader cannot answer "why does the DiT block need a gate?" or "what would break if I removed the gate terms?". They cannot modify or debug conditioning behavior in their own variants.

**Prevention:**
- Introduce the six parameters in two groups: (shift+scale for self-attn) and (gate for self-attn residual), then repeat for FFN. Show each applied separately with shape annotations.
- Demonstrate what the block computes when `gate=0` (output equals residual input — identity function). This makes the zero-initialization design choice concrete.
- Compare the Wan2.1 `DiTBlock.modulation` parameter (a learnable additive offset to the timestep projection) with the original DiT paper's adaLN-Zero, noting the subtle implementation difference.

**Warning signs:** A notebook that shows all six chunks in a single cell without separate demonstration; prose that equates "scale and shift" with "conditioning" without mentioning the gate.

**Phase:** DiTBlock notebook. Prerequisite: adaLN must be explained before patchify/unpatchify so the block forward pass makes sense end-to-end.

---

### Pitfall 4: Explaining 3D RoPE As If It Were 1D RoPE

**What goes wrong:** The tutorial introduces rotary positional embeddings from a 1D/NLP perspective ("position in the token sequence"), then shows `precompute_freqs_cis_3d` without explaining why there are three separate frequency tensors and why they cover different fractions of the head dimension.

In Wan2.1, 3D RoPE splits `head_dim` into three parts: `(dim - 2*(dim//3))` for the frame axis, `(dim//3)` for height, `(dim//3)` for width. These are then concatenated per-token after expanding across the 3D grid. A reader who doesn't understand this will think `freqs` is a flat sequence position — and will be baffled by the `torch.cat([...]).reshape(f*h*w, 1, -1)` step that builds the final frequency tensor.

**Why it happens:** Most RoPE tutorials cover 1D (LLM) RoPE. The 3D extension for video is new and under-documented in educational resources.

**Consequences:** Reader cannot explain how spatial proximity in the video tensor maps to angular proximity in frequency space. They cannot reason about what happens when video resolution changes.

**Prevention:**
- Start with 1D RoPE applied to a dummy sequence (e.g., `(B, T, D)`) to establish the rotation intuition.
- Then show how the 3D extension separates the position into three independent axes, each encoding its own positional signal in a subset of the head dimensions.
- Provide a concrete shape trace: `head_dim=128` → frame freqs cover `dim=42`, h freqs cover `dim=43`, w freqs cover `dim=43`. Show the concatenated result shape.

**Warning signs:** A notebook that only cites the 1D `precompute_freqs_cis` without separately explaining `precompute_freqs_cis_3d`; a cell that prints `freqs.shape` without explaining which axis maps to which spatial dimension.

**Phase:** SelfAttention / RoPE notebook. This must be a dedicated section, not a footnote inside the QKV notebook.

---

### Pitfall 5: Skipping the Patchify/Unpatchify Shape Contract

**What goes wrong:** A notebook shows patchify as "we slice the video into patches and flatten them into tokens." The reader understands the concept but does not understand the exact shape arithmetic: why the sequence length is `F * H_grid * W_grid`, why patch size `(1,2,2)` means spatial downsampling only (no temporal patching), and how unpatchify must receive exactly `(f, h, w)` grid dimensions to reconstruct correctly.

The Wan2.1 `WanModel.patchify` returns both the sequence and the grid tuple `(f, h, w)`. If `unpatchify` receives the wrong grid dimensions, it either crashes or silently produces a malformed video shape.

**Why it happens:** Patchify is conceptually simple, so authors rush past it. The 3D case (video vs image) adds the temporal dimension that most DiT tutorials (image-only DiT) do not cover.

**Consequences:** Readers doing exercises with modified spatial inputs will not know how to derive `f, h, w` for `unpatchify`. They will also not understand why the model works at arbitrary resolutions as long as dimensions are patch-size divisible.

**Prevention:**
- Show the full shape contract: input `(B, C, F, H, W)` → after patchify: `(B, F*(H//2)*(W//2), dim)`, where `/2` comes from patch spatial size 2.
- Demonstrate what `assert x.shape[1] == f * h * w` enforces.
- Run a deliberate failure: pass an input where `H % 2 != 0` and show the error, then explain why patch divisibility is a hard requirement.

**Warning signs:** A patchify demonstration that only shows the output shape, not the derivation from input shape; no mention of the grid dimensions tuple passed to unpatchify.

**Phase:** WanModel / patchify notebook (after DiTBlock). Address before the full model forward pass notebook.

---

### Pitfall 6: Treating CausalConv3d as a Regular Conv3d

**What goes wrong:** The VAE notebook introduces `CausalConv3d` and notes "it uses causal padding" without demonstrating what "causal" means for the temporal axis. Readers assume it behaves like a standard `nn.Conv3d` with some padding change. They miss:
- Why the temporal padding is asymmetric (2*padding on left, 0 on right): to preserve causality (no future frames leak into present frame).
- Why the `_padding` attribute stores the asymmetric values while `self.padding` is set to `(0,0,0)` (preventing PyTorch's native padding from adding symmetric padding on top).
- The streaming/cache mechanism (`cache_x`) that enables chunk-by-chunk video inference.

**Why it happens:** Causal convolution patterns are familiar from audio (WaveNet) but video causal conv with the separate cache mechanism is specialized and not widely tutorial-ized.

**Consequences:** Reader cannot explain why the VAE must process video in a specific temporal order, or why generating short clips from a model trained on longer clips requires the cache mechanism.

**Prevention:**
- Start with 1D causal convolution (a sequence kernel that looks only at past) to establish the concept.
- Then map it to the temporal axis of `CausalConv3d`: each output frame depends only on current and past input frames.
- Show the asymmetric padding explicitly: print `F.pad` arguments for a concrete kernel size, explain why left-pad length is `2 * (kernel_t - 1) / 2 = kernel_t - 1`.
- Demonstrate the cache path only if the notebook covers streaming inference; otherwise note it as out-of-scope for the architecture overview.

**Warning signs:** A notebook that shows `CausalConv3d.__init__` but not a forward pass with a concrete temporal sequence; a cell that simply says "causal padding ensures no future information leaks" without a shape-level demonstration.

**Phase:** VAE notebook. This is the most complex VAE-specific concept and should be addressed early in that notebook, before introducing ResidualBlock.

---

### Pitfall 7: Bottom-Up Order That Introduces Prerequisites Out of Sequence

**What goes wrong:** A bottom-up sequence is planned: attention → DiTBlock → WanModel → VAE. But inside attention, the RoPE frequency tensors are passed in from the outside (they are precomputed in `WanModel.__init__`, not inside `SelfAttention`). A reader studying `SelfAttention` in isolation must receive `freqs` as a parameter with no explanation of where it came from.

Similarly, `DiTBlock` receives `t_mod` (the timestep modulation tensor), which is computed in `WanModel.forward` via `time_projection`. If `DiTBlock` is taught before `WanModel`, the reader has to accept `t_mod` as "magic input from above" without the derivation.

**Why it happens:** Bottom-up ordering assumes complete internal self-containment at each level, but real architectures have cross-level dependencies that leak upward (RoPE frequencies) and downward (timestep projections).

**Consequences:** Readers get confused by unexplained parameters. They either accept them without understanding (breaks the learning goal) or get lost hunting the source (disrupts notebook flow).

**Prevention:**
- Begin each notebook with an explicit "Inputs from above" section that lists any parameters that will be explained in a later notebook, with a forward reference.
- For RoPE frequencies: create a minimal `precompute_freqs_cis` cell in the attention notebook that produces the tensor, even though the full 3D version is explained later.
- For `t_mod`: show a stub that creates a correctly shaped random tensor (`torch.randn(B, 6, dim)`) and notes "this comes from the timestep embedding — covered in the WanModel notebook."

**Warning signs:** A notebook cell that receives a parameter labeled only as "from WanModel" without a concrete shape and brief description; a notebook that cannot be run in isolation from the rest of the series.

**Phase:** Planning and sequencing phase. Must be resolved before writing Notebook 1, since the problem compounds across every subsequent notebook if not addressed at the start.

---

### Pitfall 8: Abstraction Level Mismatch Between Code and Prose

**What goes wrong:** The prose explains the architecture at a conceptual level ("the cross-attention conditions the latent on the text context"), but the code cell jumps straight to `CrossAttention.forward` internals. Or conversely, the prose is fully detailed about implementation but the code cell is a high-level API call that hides all the parts just described.

In Wan2.1, `CrossAttention` has a conditional branch: when `has_image_input=True`, the context vector `y` contains both CLIP image tokens (first 257) and text tokens (rest), and they are processed through separate key/value projections. This branch is invisible from the concept level but critical for understanding the multi-modal conditioning design.

**Why it happens:** Authors write prose to explain their mental model, then write code to demonstrate something runnable. These two things are often on different levels without the author noticing.

**Consequences:** Readers can follow either the prose or the code but cannot connect them. The conceptual understanding does not help them read the source; the source does not illuminate the concept.

**Prevention:**
- Structure each notebook section as: (1) concept in prose, (2) code cell that directly implements or calls exactly what the prose described, (3) shape printout confirming the claim.
- For branching code (like `has_image_input`): demonstrate both branches. First show `CrossAttention(has_image_input=False)` with only text context. Then show `CrossAttention(has_image_input=True)` with the `y[:, :257]` CLIP split.
- Never skip an intermediate step: if the prose describes QK computation, the next cell should show Q and K computed explicitly, not jump to the full attention output.

**Warning signs:** A prose sentence that says "the model conditions on text via cross-attention" followed by a code cell that calls the full `DiTBlock.forward` without isolating the cross-attention; prose that is shorter than two sentences for any architectural mechanism.

**Phase:** Every notebook. Reviewable as a post-writing checklist: for each prose claim, point to the exact line in the next code cell that demonstrates it.

---

### Pitfall 9: Ignoring Dtype and Device in Dummy Tensor Demonstrations

**What goes wrong:** Dummy tensors are created with default dtype (`float32`) and device (`cpu`). Some operations in Wan2.1 explicitly cast to `float64` for numerical precision (e.g., `sinusoidal_embedding_1d` uses `torch.float64`; `rope_apply` casts to `float64` for the complex multiplication). Running these cells on dummy `float32` tensors will either silently upcast or raise a dtype mismatch.

Similarly, `precompute_freqs_cis` produces complex64 tensors. `rope_apply` calls `torch.view_as_complex` which requires the last dimension to have size 2 and the tensor to be float. These constraints are not obvious from reading the function signature.

**Why it happens:** Tutorial authors focus on shapes and concepts, not dtype bookkeeping. Dtype issues feel like implementation noise rather than educational content.

**Consequences:** Cells that should demonstrate a clean shape trace instead fail with cryptic dtype errors, breaking the notebook's runnable guarantee. Or they run but produce results that cannot be traced to what real inference would do.

**Prevention:**
- Specify dtype explicitly in all dummy tensor creation: `torch.randn(..., dtype=torch.float32)`.
- When demonstrating a function that internally casts (e.g., `sinusoidal_embedding_1d`), add a cell comment explaining the cast: "internally promotes to float64 for precision, then returns in input dtype."
- For `rope_apply` and complex number operations: introduce `torch.view_as_complex` in a standalone cell before showing it used in context. Demonstrate the required `(..., 2)` last-dimension shape.

**Warning signs:** A notebook cell that creates a dummy tensor without specifying dtype; a `rope_apply` demonstration that doesn't mention complex-number representation; a cell that uses `to(torch.float64)` without explaining why.

**Phase:** RoPE / positional encoding notebook and anywhere `sinusoidal_embedding_1d` is called.

---

### Pitfall 10: Conflating the Two Patchify Functions

**What goes wrong:** The codebase contains two completely different functions named `patchify`:
- `WanModel.patchify` in `wan_video_dit.py`: uses a `Conv3d` patch embedding to convert `(B, C, F, H, W)` latents into transformer tokens.
- The module-level `patchify(x, patch_size)` in `wan_video_vae.py`: uses `einops.rearrange` to rearrange spatial pixels into channels for the VAE pixel shuffle / subpixel convolution pattern.

These perform fundamentally different operations and are used at different points in the pipeline. The DiT patchify produces transformer input tokens. The VAE patchify is a pixel-space reorganization for efficient spatial downsampling.

**Why it happens:** Both are named `patchify` because conceptually both "split a grid into patches." Readers learning bottom-up see `patchify` in the VAE notebook then encounter a different `patchify` in the DiT notebook and assume they are the same operation.

**Consequences:** Readers cannot explain why the VAE uses a rearrange-based patchify while the DiT uses a learned convolution. They conflate the two and develop incorrect intuitions about how each component processes spatial information.

**Prevention:**
- Name them explicitly in prose: "DiT patch embedding (learned convolution)" vs "VAE spatial patchification (pixel rearrangement)."
- When introducing each, immediately note the distinction: "This is different from the patchify in wan_video_vae.py — that one rearranges pixel values; this one projects them through a learned convolutional kernel."
- Show both side-by-side in a comparison cell if covering both components in the same notebook.

**Warning signs:** A notebook that uses the word "patchify" without specifying which implementation; prose that describes "the patchify operation" as if there is only one.

**Phase:** DiT patchify notebook and VAE notebook. Can be addressed with a cross-reference note in each.

---

## Moderate Pitfalls

### Pitfall 11: Not Showing What flash_attention Falls Back To

**What goes wrong:** The notebook either assumes Flash Attention is installed (imports succeed) or ignores it. In practice, most readers running notebooks locally will not have Flash Attention. The `compatibility_mode` fallback uses `F.scaled_dot_product_attention` with head-first layout, and this path works correctly — but only if demonstrated.

**Prevention:** Show both paths explicitly. Demonstrate `flash_attention(..., compatibility_mode=True)` as the "always available" baseline. Then note that Flash Attention 2/3 and SageAttn are optional performance backends. This makes the notebook runnable for everyone without installing special CUDA extensions.

**Phase:** AttentionModule notebook.

---

### Pitfall 12: Over-Annotating vs Under-Annotating Code Cells

**What goes wrong:** One extreme: every line gets an inline comment, burying the signal in noise. Other extreme: a 30-line forward pass method is pasted with no annotation at all, expecting the prose to carry the explanation load.

**Prevention:** Annotate decisions and non-obvious design choices, not syntactic steps. "# chunk into 6 modulation parameters (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)" is useful. "# return x" is not.

**Phase:** Applies to all notebooks. Establish an annotation convention in Notebook 1 and maintain it.

---

### Pitfall 13: Missing Context for the Multi-Modal Cross-Attention Design

**What goes wrong:** The notebook explains cross-attention in terms of "text conditions the latent." But Wan2.1's `CrossAttention` concatenates CLIP image tokens in front of the UMT5 text tokens when `has_image_input=True`. The hardcoded `y[:, :257]` slice (257 = 1 + 256 CLIP patch tokens) is not explained, leaving readers to wonder why 257.

**Prevention:** Introduce the full conditioning vector composition in the WanModel notebook: CLIP tokens are prepended to text tokens, making the combined context `(B, 257 + text_len, dim)`. Back-reference this when showing `CrossAttention.forward`. State that 257 = 1 class token + 16x16 CLIP spatial patch tokens.

**Phase:** CrossAttention and WanModel conditioning notebooks.

---

## Minor Pitfalls

### Pitfall 14: Gradient Checkpointing Code in Forward Pass

**What goes wrong:** The `WanModel.forward` has gradient checkpointing logic wrapped around the block loop. Including this in the educational walkthrough adds 10 lines of wrapper code that is irrelevant to understanding the architecture.

**Prevention:** Strip the gradient checkpointing branches from the walkthrough version using a simplified forward pass. Note in prose: "The production code has gradient checkpointing for training efficiency — for understanding the architecture, ignore those wrappers."

**Phase:** Full WanModel forward pass notebook.

---

### Pitfall 15: Not Stating What Is Out of Scope Per Notebook

**What goes wrong:** A reader who comes to the attention notebook wants to know about flow matching. A reader in the DiT block notebook asks about LoRA. The notebook silently ignores these or the author adds tangential asides that derail the focus.

**Prevention:** Include a one-cell "Out of scope for this notebook" section near the top listing topics deliberately excluded (e.g., "Training, flow matching loss, LoRA injection are not covered here — see the other course materials"). This resets reader expectations before they get confused.

**Phase:** All notebooks.

---

## Phase-Specific Warnings

| Phase / Notebook Topic | Likely Pitfall | Mitigation |
|------------------------|----------------|------------|
| Attention + QKV | Shape convention (b s n d vs b n s d), flash attn backends | Pitfall 2, 11 — show both layouts and both execution paths |
| RMSNorm + SelfAttention | RoPE frequencies as magic input from outside | Pitfall 7 — include forward-reference stub for freqs |
| 3D RoPE | 1D-only mental model, complex number dtype | Pitfall 4, 9 — standalone 1D → 3D progression with dtype trace |
| DiTBlock | adaLN six-parameter confusion, missing gate intuition | Pitfall 3 — demonstrate identity behavior at gate=0 |
| WanModel patchify/unpatchify | Wrong dummy shapes, conflating two patchify functions | Pitfall 1, 5, 10 — derive shapes from config constants |
| VAE CausalConv3d | Treating as standard Conv3d | Pitfall 6 — asymmetric padding cell, skip cache mechanism |
| VAE patchify/unpatchify | Conflation with DiT patchify | Pitfall 10 — explicit naming distinction |
| Full WanModel forward | Gradient checkpointing clutter, multi-modal conditioning 257 split | Pitfall 13, 14 — strip training branches, explain CLIP token count |
| Any notebook | Prose/code abstraction mismatch, over/under annotation | Pitfall 8, 12 — review checklist per section |

---

## Sources

- Wan2.1 source: `diffsynth/models/wan_video_dit.py`, `diffsynth/models/wan_video_vae.py` (direct code inspection, HIGH confidence)
- DiT paper architecture: [Diffusion Transformer Explained — Towards Data Science](https://towardsdatascience.com/diffusion-transformer-explained-e603c4770f7e/) (MEDIUM confidence, summarized by fetch)
- adaLN-Zero mechanics: [DiT/models.py — facebookresearch/DiT](https://github.com/facebookresearch/DiT/blob/main/models.py) and [AdaLN-Zero Conditioning — DeepWiki](https://deepwiki.com/sontungkieu/shortcut-models/5.3-adaln-zero-conditioning) (MEDIUM confidence)
- RoPE and 3D extension: [Rotary Embeddings — EleutherAI Blog](https://blog.eleuther.ai/rotary-embeddings/) and [RoPE, Clearly Explained — TDS](https://towardsdatascience.com/rope-clearly-explained/) (MEDIUM confidence)
- Video VAE causal convolution: [Improved Video VAE for Latent Video Diffusion Model — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Improved_Video_VAE_for_Latent_Video_Diffusion_Model_CVPR_2025_paper.pdf) (MEDIUM confidence)
- Common PyTorch mistakes: [Most Common Neural Net PyTorch Mistakes — Medium](https://medium.com/missinglink-deep-learning-platform/most-common-neural-net-pytorch-mistakes-456560ada037) (MEDIUM confidence)
- UvA Deep Learning Tutorials (educational notebook design reference): [uvadlc-notebooks.readthedocs.io](https://uvadlc-notebooks.readthedocs.io/) (MEDIUM confidence)
- einops conventions: [Einops tutorial — einops.rocks](https://einops.rocks/1-einops-basics/) (HIGH confidence)
