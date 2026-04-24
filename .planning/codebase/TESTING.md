# Testing Patterns

**Analysis Date:** 2026-04-24

## Test Framework

**Runner:**
- No formal test framework detected
- Ad-hoc test script present: `test_files/test_pt_files.py`
- Manual verification approach via inspection scripts

**Assertion Library:**
- Python built-in `assert` statements used for validation in core modules
- No dedicated testing library (pytest, unittest) configured

**Run Commands:**
```bash
python test_files/test_pt_files.py  # Inspect cached tensor files
python inference.py                  # Manual inference test
python fast_inference.py             # Cache-based inference test
```

## Test File Organization

**Location:**
- Test files in dedicated `test_files/` directory
- Ad-hoc approach; not co-located with source code
- Single test file present: `test_files/test_pt_files.py`

**Naming:**
- Pattern: `test_*.py` for test files
- File: `test_pt_files.py` - inspects PyTorch serialized tensor files

**Structure:**
```
test_files/
└── test_pt_files.py              # Tensor inspection utility
```

## Test Structure

**Test File Pattern:**
```python
# test_pt_files.py - Inspection-based testing
import torch
import os

def print_structure(files_path):
    """Inspect a .pt file and print its structure."""
    data = torch.load(files_path, map_location="cpu")
    print(f"Type: {type(data)}")
    if isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key} | Shape: {value.shape} | Dtype: {value.dtype}")
            else:
                print(f"Key: {key} | Type: {type(value)}")
    # ... etc

if __name__ == "__main__":
    files_path = "/path/to/cache/sample_000000"
    pt_files = [file for file in os.listdir(files_path) if file.endswith('.pt')]
    for pt_file in pt_files:
        print(f"\nInspecting file: {pt_file}")
        print_structure(os.path.join(files_path, pt_file))
```

**Patterns:**
- Main entry point: `if __name__ == "__main__":`
- Functions focused on inspection/validation: `print_structure()`
- Type checking: `isinstance()` for distinguishing tensor types
- Iteration with print progress: Print per-file inspection results

## Validation Approach

**Unit Validation:**
- File existence checks before loading: `if not sample_path.exists():`
- Tensor shape validation: `assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"`
- Type validation with isinstance: `if isinstance(data, torch.Tensor):`

**Data Integrity:**
- Cache index validation: Load and verify `cache_index.json` structure
- Tensor format validation: Check dtype and shape of loaded tensors
- File completeness checks: Verify optional files exist before loading

**Examples from codebase:**
```python
# From train_lora_cached.py - CachedLatentDataset
def __getitem__(self, idx):
    # Load required tensors
    data["input_latents"] = torch.load(..., weights_only=True)
    data["context"] = torch.load(..., weights_only=True)
    
    # Load optional tensors with existence checks
    if (sample_path / "control_latents.pt").exists():
        data["control_latents"] = torch.load(...)
    if (sample_path / "reference_latents.pt").exists():
        data["reference_latents"] = torch.load(...)
    
    # Load metadata
    with open(sample_path / "metadata.json", "r") as f:
        data["metadata"] = json.load(f)
    
    return data

# From general_modules.py - Shape assertion
def get_timestep_embedding(...):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    # ... processing
```

## Integration Testing

**Script-Based Testing:**
- `inference.py` - Full pipeline inference test with LoRA loading
- `fast_inference.py` - Cached inference test with optional T5/CLIP loading
- `train_lora_cached.py` - Training pipeline test via argparse execution
- `preprocess_cache.py` - Caching pipeline test with dataset creation

**Manual Test Procedures:**
1. Cache preprocessing: Run `preprocess_cache.py` with dataset
2. Training validation: Run `train_lora_cached.py` with cached data
3. Inference test: Run `inference.py` or `fast_inference.py` with test data
4. Tensor inspection: Run `test_pt_files.py` to verify cache contents

**Test Data:**
- Existing cache directories: `./cache/`, `./cache/fast_inference/`
- Sample data: `./data/` subdirectories with video and reference images
- Output directories: `./output/`, `./results/` for generated artifacts

## Mocking

**Framework:** No formal mocking framework (unittest.mock) detected

**Patterns:**
- Cache file stubbing in `fast_inference.py`: Check for cached tensors before loading models
- Conditional model loading based on cache state:
  ```python
  use_cache = cache_exists()
  if not use_cache:
      # Load full models for computation
      model_configs += [T5_config, CLIP_config]
  else:
      # Skip expensive model loading
      print("Cache found — skipping T5 + CLIP model loading")
  ```

**What to Mock:**
- Model loading: Stub model downloads/loading for unit tests
- Data file I/O: Mock filesystem operations for cache tests
- CUDA operations: Mock torch.cuda for CPU-only testing

**What NOT to Mock:**
- PyTorch operations: Use actual tensor operations for correctness
- Pipeline units: Test actual unit composition and forwarding
- Data loading: Validate actual dataset class behavior with real data

## Fixtures and Factories

**Test Data:**
- Cached data directory structure: `cache/{sample_id}/` with `.pt` files and `metadata.json`
- Sample directory: `cache/sample_000000/` with serialized tensors
- Metadata fixture: `metadata.json` files in sample directories

**Location:**
- Sample cache: `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/cache/`
- Test reference data: `./data/` directory with video and image files
- Generated cache: Created dynamically by `preprocess_cache.py`

**Data Structure:**
```python
# Cache directory structure (from CachedLatentDataset)
cache/
├── cache_index.json                    # Metadata for all samples
└── sample_000000/
    ├── input_latents.pt                # VAE-encoded target video
    ├── context.pt                      # Text encoder output
    ├── control_latents.pt              # (optional) VAE control video
    ├── reference_latents.pt            # (optional) VAE reference image
    ├── clip_feature.pt                 # (optional) CLIP reference encoding
    └── metadata.json                   # Height, width, num_frames, prompt
```

## Coverage

**Requirements:** None enforced; ad-hoc testing approach

**Observation:**
- Core model classes use assertions for type checking
- Data loading validated through isinstance checks and file existence checks
- Training pipeline tested via script execution with real data
- No automated coverage measurement

## Test Types

**Inspection Tests:**
- Purpose: Verify cached data structure and content
- Approach: Load tensors and print shape/dtype/keys
- Example: `test_pt_files.py` inspects `.pt` files in cache

**End-to-End Integration Tests:**
- Purpose: Validate full pipeline execution
- Approach: Run complete scripts with test data
- Examples:
  - `preprocess_cache.py` with test dataset
  - `train_lora_cached.py` with cached data
  - `inference.py` with control video and reference image
  - `fast_inference.py` with cached embeddings

**Unit Validation:**
- Purpose: Ensure individual functions handle edge cases
- Approach: Assertions and type checks embedded in functions
- Examples:
  - `assert len(timesteps.shape) == 1` in timestep embedding
  - Existence checks before file loading
  - Optional tensor handling with fallback to zeros

**No formal test suite:** Testing primarily manual and script-based

## Common Patterns

**Async/Concurrency Testing:**
- Not applicable; codebase uses synchronous execution
- PyTorch DataLoader used for batching, not async operations

**Error Testing:**
- Exception raises validated by function caller
- File not found errors raised with descriptive messages
- Invalid input types caught with isinstance checks

**Example Error Handling Pattern:**
```python
# From dataset_collection/prepare_dataset.py
def init_face_model(model_dir, device):
    """Load FLAME model with validation."""
    config = make_config(model_dir)
    for attr in ("flame_model_path", "static_landmark_embedding_path"):
        p = getattr(config, attr)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Download from https://flame.is.tue.mpg.de/"
            )
    # ... continue with safe file operations
```

## Test Execution

**Manual Script Testing:**
```bash
# Inspect cache structure
python test_files/test_pt_files.py

# Full preprocessing pipeline
python preprocess_cache.py \
    --dataset_base_path ./data/video_dataset \
    --output_path ./cache

# Training with cached data
python train_lora_cached.py \
    --cache_path ./cache \
    --output_path ./lora_output \
    --num_epochs 5

# Inference test
python inference.py

# Fast inference with cache
python fast_inference.py
```

**Success Criteria:**
- Cache files created with expected structure (input_latents.pt, context.pt, etc.)
- Training completes without OOM errors
- Inference generates valid video output
- Tensor inspection shows correct shapes and dtypes

## Known Test Gaps

**Areas with Limited Coverage:**
- Edge cases in dataset loading (corrupted files, incomplete metadata)
- Error recovery in training (checkpoint resumption, partial data)
- Multi-GPU/distributed training validation
- Memory management under constrained resources

**Recommended Future Tests:**
- Unittest suite for dataset classes with fixture data
- Integration tests for preprocessing → training → inference pipeline
- Pytest fixtures for repeatable test data
- Parametrized tests for different model configurations

---

*Testing analysis: 2026-04-24*
