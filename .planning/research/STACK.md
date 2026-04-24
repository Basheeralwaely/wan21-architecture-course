# Technology Stack

**Project:** Wan2.1 Model Architecture Course
**Researched:** 2026-04-24
**Scope:** Educational Jupyter notebooks teaching DiT/VAE architecture from `diffsynth/` source code

---

## Recommended Stack

### Notebook Runtime

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| JupyterLab | >=4.3 | Authoring and running notebooks | Standard environment for interactive ML notebooks. JupyterLab 4.x is the current-generation interface (replaces classic Notebook UI). `.ipynb` format (nbformat 4) is the universal interchange format — VS Code, Colab, JupyterLab all open it natively. |
| nbformat | 5.10 | Notebook file format | Current stable format. nbformat 4 minor versions are backward-compatible. Use `"nbformat": 4, "nbformat_minor": 5` in notebook metadata. |
| Python | 3.10+ | Kernel language | Matches existing codebase environment (confirmed Python 3.10.18 in this repo). PyTorch 2.8+ requires 3.10+. |

### Core Computation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | >=2.8.0 | All tensor operations, dummy tensor construction, forward pass tracing | Already installed in this environment (2.8.0+cu126 confirmed). The notebooks import directly from `diffsynth/models/` which depends on PyTorch. Pinning >=2.8 gives access to `F.scaled_dot_product_attention` (used by the DiT's fallback attention path) and stable RoPE support. Do not require GPU — all demo tensors run on CPU with `torch.zeros` / `torch.randn`. |
| einops | >=0.8.1 | Tensor rearrangement annotations | Already installed (0.8.1 confirmed). The DiT source code (`wan_video_dit.py`) uses `einops.rearrange` pervasively for QKV head-splitting. Notebooks should import einops and demonstrate `rearrange("b s (n d) -> b n s d", n=heads)` patterns explicitly — it is central to understanding multi-head attention layout. |
| NumPy | >=2.2.5 | Numerical helpers, array display | Already installed (2.2.5 confirmed). Needed for printing weight stats and simple position embedding math. |

### Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Matplotlib | >=3.10.3 | Architecture diagrams, attention weight heatmaps, tensor shape illustrations | Already installed (3.10.3 confirmed). The de-facto standard for static visualizations in educational notebooks. Use `imshow` for attention maps, `bar` for parameter counts by layer. No interactivity needed for architecture tutorials — static SVG-quality plots are sufficient and reproducible. |
| torchinfo | 1.8.0 | Model summary tables (parameter counts, input/output shapes per layer) | The key display tool for architecture education. `summary(model, input_data=x, col_names=["input_size","output_size","num_params","trainable"])` gives the hierarchy table that answers "what does this module do to tensor shapes?" without running full inference. Replaces tedious `print(model)`. Not installed yet — add to notebook requirements cell. |

### Supporting Utilities

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ipywidgets | >=8.1.8 | Interactive sliders for exploring hyperparameter effects (e.g., varying `num_heads` or `hidden_dim`) | Optional but valuable for the attention notebook specifically — a slider showing how head dimension = hidden_dim / num_heads changes shapes. Do not rely on it for core content; make every cell also work as static output. |
| tqdm | >=4.66.0 | Progress display in multi-step cells | Already in requirements.txt. Use sparingly — only in cells that iterate over many timesteps or layers to illustrate something. |

---

## What NOT to Use and Why

| Tool | Why Avoid |
|------|-----------|
| **PyTorch Lightning / Accelerate** | Training frameworks — out of scope for architecture-only notebooks. Would add import weight and conceptual noise. |
| **Plotly / Bokeh** | Interactive web-based plots require extra frontend setup and don't render cleanly in static nbconvert exports. Matplotlib is sufficient and always renders. |
| **torchviz / graphviz** | Computational graph visualization tools are notoriously brittle (require system graphviz binary, produce unreadable graphs for large models like the 30-block DiT). Use torchinfo tables instead. |
| **JAX / TensorFlow** | The source code is PyTorch — never mix frameworks in tutorial notebooks. |
| **transformers / PEFT** | Training dependencies. Architecture notebooks don't need them — they import directly from `diffsynth/models/`. |
| **Flash Attention / SageAttention** | Optional C-extension backends. Notebooks must run on CPU without these. The DiT source already has a CPU fallback via `F.scaled_dot_product_attention` — notebooks should explicitly use that path with a comment explaining the production alternatives. |
| **JupyterLite / Colab-specific APIs** | Keep notebooks environment-agnostic. No `google.colab.drive` mounts, no `!pip install` cells that assume Colab. Use a `requirements_course.txt` instead. |

---

## Notebook Conventions (Style Stack)

These are not libraries but conventions that constitute the "authoring stack" — the UvA Deep Learning Tutorials and Annotated Transformer are the reference style.

### Cell Organization Pattern (per notebook)

```
[Markdown] Title + Learning Objectives (H1, bullet list of what reader learns)
[Markdown] Prerequisites + Setup note
[Code]     Imports cell (all imports at top, grouped: stdlib, torch, einops, matplotlib, then diffsynth)
[Markdown] Section H2: Concept introduction
[Code]     Minimal construction of component with dummy tensors
[Code]     Shape assertion cell: assert output.shape == expected, f"got {output.shape}"
[Markdown] Section H2: Next concept
... repeat
[Markdown] Summary + what's next
```

### Dummy Tensor Convention

```python
# Always annotate what dimensions represent
B, T, H, W = 1, 9, 64, 64        # batch, frames, spatial height, spatial width
C = 16                             # latent channels (VAE output)
x = torch.randn(B, C, T, H, W)   # (batch, channels, frames, height, width)
```

Use `torch.randn` for continuous features, `torch.zeros` when only shapes matter, `torch.randint` for discrete indices. Always state shapes in comments. This is the pattern used by the official PyTorch tutorials and d2l.ai.

### Shape Verification Cells

Every forward pass cell should end with:
```python
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
assert out.shape == (B, T_out, D), f"Unexpected shape: {out.shape}"
```

This teaches shape awareness and surfaces breakage immediately without requiring pytest.

---

## Installation

The environment already has PyTorch, einops, matplotlib, numpy. Only two additions are needed:

```bash
# For notebooks only (not part of training pipeline)
pip install jupyterlab>=4.3 torchinfo>=1.8.0 ipywidgets>=8.1.8
```

Or create `Course/requirements_course.txt`:
```
# Existing in repo environment
torch>=2.8.0
einops>=0.8.0
matplotlib>=3.10.0
numpy>=2.0.0

# New additions for Course notebooks
jupyterlab>=4.3
torchinfo>=1.8.0
ipywidgets>=8.1.8
```

No GPU is required. All cells use CPU tensors with `device="cpu"` explicitly or by default.

---

## Confidence Assessment

| Component | Confidence | Basis |
|-----------|------------|-------|
| PyTorch 2.8 as baseline | HIGH | Confirmed installed in this environment; version from `pip list` |
| einops as core visualization tool for attention | HIGH | einops 0.8.1 confirmed installed; source code uses it throughout |
| Matplotlib 3.10 | HIGH | Confirmed installed (3.10.3); stable release |
| torchinfo 1.8.0 | HIGH | Confirmed latest PyPI release; well-maintained (87k downloads/month per Libraries.io) |
| JupyterLab 4.3 | MEDIUM | Version inferred from 2025 JupyterLab release cadence; not installed yet — verify latest stable at install time |
| ipywidgets 8.1.8 | MEDIUM | Confirmed latest PyPI release; listed as optional enhancement |
| Notebook style conventions | MEDIUM | Based on survey of UvA DL Notebooks, Annotated Transformer, d2l.ai — all convergent on same pattern |

---

## Sources

- PyTorch stable release: https://pypi.org/project/torch/ (2.11.0 as of 2026-03-23; 2.8.0 in this environment)
- torchinfo: https://github.com/TylerYep/torchinfo (v1.8.0 confirmed latest)
- Matplotlib releases: https://matplotlib.org/stable/users/release_notes.html (3.10.8 as of 2025-12-10)
- ipywidgets: https://ipywidgets.readthedocs.io/en/latest/ (8.1.8 as of 2025-11-01)
- einops: https://pypi.org/project/einops/ (0.8.2 as of 2026-01-26; 0.8.1 installed here)
- nbformat specification: https://nbformat.readthedocs.io/en/latest/format_description.html
- Reference style: UvA Deep Learning Tutorials https://uvadlc-notebooks.readthedocs.io/
- Reference style: The Annotated Transformer https://github.com/harvardnlp/annotated-transformer
- Reference style: Dive into Deep Learning https://d2l.ai/
