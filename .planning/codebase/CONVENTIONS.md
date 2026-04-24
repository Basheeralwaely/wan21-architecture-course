# Coding Conventions

**Analysis Date:** 2026-04-24

## Naming Patterns

**Files:**
- Module files use snake_case: `train_lora_cached.py`, `preprocess_cache.py`, `fast_inference.py`
- Class-based modules use descriptive names: `wan_video_dit.py`, `wan_video_vae.py`, `general_modules.py`
- Entry point scripts are descriptive action names: `inference.py`, `train_full_pipeline.py`

**Functions:**
- Functions use snake_case: `compute_flow_match_loss()`, `prepare_y_tensor()`, `encode_prompt()`
- Internal/helper functions use leading underscore where appropriate (sparse usage observed)
- Factory/builder functions are explicit: `create_cache_dataset()`, `load_models_for_caching()`
- Getter functions use `load_` or `fetch_` prefix: `load_dit_for_training()`, `load_cache()`
- Processing functions use `compute_` or `encode_` prefix: `compute_flow_match_loss()`, `encode_sample()`

**Classes:**
- Classes use PascalCase: `CachedLatentDataset`, `TimestepEmbeddings`, `WanVideoPipeline`, `RMSNorm`
- Pipeline/Unit classes are explicit: `WanVideoUnit_PromptEmbedder`, `WanVideoPostUnit_S2V()`, `BasePipeline`
- Model classes follow `WanModel*` pattern: `WanModel`, `WanTextEncoder`, `WanVideoVAE`, `WanMotionControllerModel`
- Normalization classes use full names: `RMSNorm`, `AdaLayerNorm`, `TemporalTimesteps`

**Variables:**
- Local variables use snake_case: `input_latents`, `batch_size`, `num_frames`, `device`
- Tensor variables are descriptive: `noisy_latents`, `target`, `context`, `clip_feature`
- Boolean flags use `is_` or `use_` prefix: `use_gradient_checkpointing`, `use_cache`
- Config dictionaries use abbreviated names: `y`, `t`, `emb` (timestep embeddings)
- Batch dictionary keys use snake_case: `"input_latents"`, `"control_latents"`, `"reference_latents"`, `"metadata"`

**Types:**
- Type hints use Python typing module: `Union[str, torch.device]`, `Optional[torch.Tensor]`, `Literal[...]`
- Device types use string literals: `"cuda"`, `"cpu"`
- Data types reference PyTorch dtypes: `torch.bfloat16`, `torch.float32`

## Code Style

**Formatting:**
- No explicit formatter configured (Ruff/Black not detected)
- Indentation: 4 spaces (observed consistently)
- Line length: No strict limit observed; ranges 80-120+ characters
- Import spacing: One blank line between imports and code
- Function spacing: Two blank lines between top-level functions

**Linting:**
- No linting config files detected (`.flake8`, `pylintrc` absent)
- Code follows implicit Python conventions
- Assertion-based validation preferred: `assert len(timesteps.shape) == 1`

## Import Organization

**Order:**
1. Standard library imports: `torch`, `os`, `argparse`, `json`, `math`, `time`
2. Third-party ML frameworks: `torch`, `torch.nn`, `accelerate`, `transformers`, `peft`
3. Data/image processing: `numpy`, `pandas`, `PIL`, `imageio`, `tqdm`
4. Internal diffsynth imports: `from diffsynth.pipelines...`, `from diffsynth.models...`
5. Local relative imports: `from .models import ...`, `from .config import ...`

**Path Aliases:**
- Relative imports use dot notation: `from ..core import ModelConfig`, `from ..models.wan_video_dit import WanModel`
- Two-dot relative imports traverse package hierarchy: `from ..diffusion import FlowMatchScheduler`
- Module imports are explicit: `from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig`

**Examples:**
```python
# train_lora_cached.py
import torch
import torch.nn as nn
import os
import argparse
import json
import accelerate
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import FlowMatchScheduler
```

## Error Handling

**Patterns:**
- Explicit exception types raised: `raise FileNotFoundError()`, `raise ValueError()`, `raise RuntimeError()`
- Validation with assertions: `assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"`
- Try-except for file operations: `try: ... except Exception as e:` for data loading operations
- File existence checks: `if not sample_path.exists():` before attempting load
- Error messages are descriptive with context: `raise FileNotFoundError(f"Missing model file: {p}\nDownload from https://...")`

**Examples from codebase:**
```python
# File validation
if not os.path.isfile(p):
    raise FileNotFoundError(f"Missing model file: {p}\nDownload from https://flame.is.tue.mpg.de/")

# Exception handling in data loading
try:
    data = torch.load(sample_path / "input_latents.pt", weights_only=True)
except Exception as e:
    # Handle error
    pass

# Runtime validation
assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
```

## Logging

**Framework:** Console output via `print()` - no dedicated logging framework used

**Patterns:**
- Progress tracking with `tqdm`: `pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")`
- Informational prints: `print(f"Model loaded in {elapsed:.2f} seconds.")`
- Status updates: `print(f"Dataset size: {len(dataset)} (with {repeat}x repeat)")`
- Formatted output with f-strings: `print(f"Trainable parameters: {num:,} / {total:,} ({pct:.2f}%)")`
- File operation confirmation: `print(f"Saved {key} → {path} {shape}")`

**When to log:**
- Model loading/initialization: "Model loaded in X seconds"
- Data loading completion: "Dataset size: N samples"
- Training checkpoints: "Saved checkpoint: /path/to/checkpoint"
- File operations: "Saved X to path"
- Epoch/batch progression: Using tqdm for loop iteration

**Not logged:**
- Debug-level iteration details (individual batch processing)
- Memory usage (handled by accelerate framework)
- Intermediate tensor values

## Comments

**When to Comment:**
- Module docstrings present for most scripts: explain purpose and usage in docstring
- Class docstrings present: `"""Load FLAME model, face-only triangle mask, and MediaPipe embedding."""`
- Function docstrings for non-obvious functions: `"""Encode a single prompt with T5, matching WanVideoUnit_PromptEmbedder."""`
- Inline comments for complex logic: `# Target is the velocity field: noise - data (matches scheduler.training_target convention)`
- Section headers for logical groupings: `# ── Configuration ──────────────────────────────────────────────────────────`

**JSDoc/TSDoc:**
- Not applicable; codebase is Python (no TypeScript)
- Docstrings use triple-quote format: `"""Docstring here."""`
- Multi-line docstrings for complex functions present in training scripts

**Examples:**
```python
# From train_lora_cached.py - docstring
"""
Step 2: Train LoRA with Cached Data
====================================
This script trains LoRA on pre-cached latents/embeddings.
Much faster than train_full_pipeline.py because:
- No VAE encoding (expensive!)
- No text encoding
- No CLIP encoding
- Only DiT forward/backward passes

Usage:
1. First run: python preprocess_cache.py --dataset_base_path ... --output_path ./cache
2. Then run: python train_lora_cached.py --cache_path ./cache --output_path ./lora_output
"""

# From general_modules.py - inline comment
# concat sine and cosine embeddings
emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
```

## Function Design

**Size:** Functions range 5-50 lines typically; larger functions (100+ lines) are data loading/inference pipelines

**Parameters:**
- Most functions accept explicit parameters rather than config objects
- Device and dtype passed explicitly: `device`, `dtype`, `torch_dtype=torch.bfloat16`
- Optional parameters use Python defaults: `repeat=1`, `weights_only=True`
- Batch operations accept dictionary inputs: `batch["input_latents"]`, `batch["context"]`

**Return Values:**
- Single returns typical: `return model`, `return cache`, `return embeddings`
- Multiple related returns use tuple unpacking: `y, reference_latents = prepare_y_tensor(batch, dit, device, dtype)`
- Dictionary returns for grouped data: `return {"latents": ..., "metadata": ...}`
- None returns for side-effect operations: `save_cache()` returns None

**Examples:**
```python
# Simple return
def get_trainable_params(model):
    """Get only LoRA parameters for training."""
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return trainable_params

# Multiple returns
def prepare_y_tensor(batch, dit, device, dtype):
    """Prepare the y conditioning tensor from cached data."""
    # ... processing ...
    return y, reference_latents

# Dictionary return
def encode_sample(pipe, data, device, extra_inputs):
    """Encode a single sample and return cached tensors."""
    cache = {}
    cache["metadata"] = {"height": ..., "width": ..., "num_frames": ...}
    # ... more setup ...
    return cache
```

## Module Design

**Exports:**
- Classes are exported directly: `WanVideoPipeline`, `CachedLatentDataset`, `TemporalTimesteps`
- Main functions exposed at module level: `def train_epoch()`, `def setup_lora()`
- Internal functions use descriptive names but no underscore prefix convention observed

**Barrel Files:**
- `__init__.py` files present in packages: `diffsynth/__init__.py`, `diffsynth/configs/__init__.py`
- Barrel files used for API exposure in some packages
- Direct imports from submodules preferred: `from diffsynth.pipelines.wan_video import WanVideoPipeline`

**Package Structure:**
- Logical organization by component: `models/`, `pipelines/`, `configs/`, `core/`, `utils/`
- Dataset classes co-located with training scripts: `CachedLatentDataset` in `train_lora_cached.py`
- Utility functions in `utils/` subdirectory: `diffsynth/utils/data/__init__.py`

## Device and Data Type Handling

**Explicit Type Casting:**
- Data type conversions explicit: `.to(dtype=torch.bfloat16)`, `.float()`, `.to(torch.float32)`
- Device placement explicit: `.to(device)`, `.to(accelerator.device)`
- Dtype preserved across operations: `dtype=latents.dtype` when creating matching tensors

**Batch Processing:**
- Batch items accessed via dictionary keys: `batch["input_latents"]`, `batch["metadata"]`
- Collate functions handle batching: `def collate_fn(batch):` concatenates tensors
- Gradient checkpointing configuration passed through: `use_gradient_checkpointing=False`

---

*Convention analysis: 2026-04-24*
