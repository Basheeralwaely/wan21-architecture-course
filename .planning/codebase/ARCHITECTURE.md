# Architecture

**Analysis Date:** 2026-04-24

## Pattern Overview

**Overall:** Two-Stage Diffusion Model Training Pipeline

**Key Characteristics:**
- **Modular separation** between expensive preprocessing and fast training via caching
- **Flow matching diffusion** framework with step-wise denoising
- **LoRA adaptation** for efficient model fine-tuning on custom video generation
- **Hardware-optimized** with gradient checkpointing and memory offloading options
- **Multi-modal conditioning** (text, control video, reference image, CLIP embeddings)

## Layers

**Entry Point Layer:**
- Purpose: User-facing training scripts and inference interfaces
- Location: `train_full_pipeline.py`, `train_lora_cached.py`, `preprocess_cache.py`, `inference.py`
- Contains: CLI argument parsing, pipeline orchestration, training loops
- Depends on: DiffSynth pipeline, model loading utilities
- Used by: Command line execution, shell wrapper scripts

**Pipeline Layer:**
- Purpose: Orchestrates the full generation and training workflow
- Location: `diffsynth/pipelines/wan_video.py`
- Contains: `WanVideoPipeline` class with unit runners, scheduler configuration
- Depends on: Model components (DiT, VAE, text encoders), diffusion scheduler, data operators
- Used by: Training modules, inference scripts

**Model Components Layer:**
- Purpose: Individual neural network models that perform specific encoding/decoding tasks
- Location: `diffsynth/models/`
- Contains: 
  - `wan_video_dit.py`: Diffusion Transformer (DiT) - 30-block core denoising model (~780M params)
  - `wan_video_vae.py`: Video Autoencoder for latent encoding/decoding
  - `wan_video_text_encoder.py`: T5/UMT5 text embedding encoder
  - `wan_video_image_encoder.py`: CLIP vision encoder for reference image embedding
  - Motion controllers, animate adapters, audio encoders for specialized conditioning
- Depends on: PyTorch, Hugging Face transformers
- Used by: Pipeline units and training modules

**Data Loading Layer:**
- Purpose: Flexible dataset handling with operator composition
- Location: `diffsynth/core/data/`
- Contains: `UnifiedDataset`, video/image loaders, operator chains
- Depends on: PIL, OpenCV, torchvision
- Used by: Training scripts, preprocessing

**Training Framework Layer:**
- Purpose: Distributed training harness and gradient optimization
- Location: `diffsynth/diffusion/`
- Contains:
  - `training_module.py`: Base `DiffusionTrainingModule` with forward pass
  - `loss.py`: Flow matching loss computation
  - `flow_match.py`: Flow matching scheduler
  - `runner.py`: Distributed training launcher
  - `logger.py`: Checkpoint saving utilities
- Depends on: Accelerate, PyTorch distributed, PEFT (LoRA)
- Used by: Training scripts

**Caching Layer:**
- Purpose: Pre-compute expensive model encodings
- Location: Standalone preprocessor `preprocess_cache.py` outputs to `cache/` directory
- Contains: JSON index, `.pt` tensor files organized by sample
- Depends on: All model components
- Used by: `train_lora_cached.py` for fast training iterations

**Utility Layers:**
- **Core utilities** (`diffsynth/core/`): Attention mechanisms, gradient checkpointing, device management, model loading
- **LoRA utilities** (`diffsynth/utils/lora/`): PEFT adapter injection
- **Control utilities** (`diffsynth/utils/controlnet/`): ControlNet input preparation

## Data Flow

**Pipeline A: Full Training (train_full_pipeline.py)**

1. **Data Loading**: UnifiedDataset loads video, control video, reference image, prompt from CSV metadata
2. **Encoding (per epoch)**: 
   - VAE encodes video frames → input latents
   - Text encoder encodes prompt → context
   - VAE encodes control video → control latents
   - CLIP encodes reference image → clip features, VAE encodes reference → reference latents
3. **Model Forward Pass**: WanVideoPipeline units process loaded tensors
   - Concatenate control + reference latents to form conditioning (y)
   - DiT denoises noisy latents using context and conditioning across 50 timesteps
4. **Loss Computation**: Flow matching loss (MSE between predicted and target velocity)
5. **Gradient Update**: Backward pass, optimizer step (only LoRA params trainable)

**Pipeline B: Cached Training (train_lora_cached.py) - Recommended**

1. **Step 1 - Preprocess (preprocess_cache.py)**: ONE-TIME
   - Load full pipeline with all encoders (VAE, text, CLIP)
   - Iterate dataset, encode each sample independently
   - Save to disk: input_latents, context, control_latents, reference_latents, clip_feature, metadata
   - Create JSON index for dataset tracking
2. **Step 2 - Training (train_lora_cached.py)**: REPEATABLE
   - Load only DiT model (no VAE, encoders needed)
   - CachedLatentDataset loads pre-computed tensors from disk
   - Sample random timestep, random noise
   - DiT processes noisy latents + context + conditioning
   - Compute flow matching loss
   - Gradient update (only LoRA trainable)

**State Management:**
- **Model state**: Loaded once, LoRA adapters injected via PEFT
- **Training state**: Optimizer, scheduler, loss accumulation per batch
- **Caching state**: Persistent disk storage (`cache/`) indexed by JSON
- **Output state**: LoRA checkpoints saved to `output/` as SafeTensor files

## Key Abstractions

**WanVideoPipeline:**
- Purpose: Composes all model components and defines execution units
- Examples: `diffsynth/pipelines/wan_video.py` lines 32-82
- Pattern: Pipeline orchestrator with unit-based architecture

**DiffusionTrainingModule:**
- Purpose: Base training class handling forward pass, loss computation, distributed setup
- Examples: `train_full_pipeline.py` lines 9-108
- Pattern: Inheritance-based training harness with task-specific loss functions

**UnifiedDataset:**
- Purpose: Flexible dataset with composable operators for loading and transforming data
- Examples: `diffsynth/core/data/unified_dataset.py` lines 5-58
- Pattern: Operator composition via `>>` chain syntax (e.g., `ToAbsolutePath() >> LoadVideo() >> ImageCropAndResize()`)

**CachedLatentDataset:**
- Purpose: Lightweight dataset that loads pre-cached PyTorch tensors
- Examples: `train_lora_cached.py` lines 32-70
- Pattern: Direct tensor loading with minimal preprocessing

**LoRA Injection:**
- Purpose: Selective parameter adaptation using PEFT library
- Examples: `train_lora_cached.py` lines 101-125
- Pattern: QKV/FFN layer targeting with rank-based factorization

## Entry Points

**train_full_pipeline.py:**
- Location: `train_full_pipeline.py` lines 123-185
- Triggers: `python train_full_pipeline.py [args]`
- Responsibilities: 
  - Parse CLI arguments (dataset, model, learning rate, LoRA config)
  - Create UnifiedDataset with full operator chain
  - Instantiate WanTrainingModule with full model loading
  - Launch distributed training via accelerate

**train_lora_cached.py:**
- Location: `train_lora_cached.py` lines 287-387
- Triggers: `python train_lora_cached.py [args]` or `accelerate launch train_lora_cached.py [args]`
- Responsibilities:
  - Load only DiT from cache
  - Create CachedLatentDataset from preprocessed files
  - Setup LoRA adapters with target modules
  - Launch training epochs with batched loss computation

**preprocess_cache.py:**
- Location: `preprocess_cache.py` lines 163-217
- Triggers: `python preprocess_cache.py [args]` or `./step1_cache_data.sh`
- Responsibilities:
  - Load full pipeline (VAE, text encoder, CLIP)
  - Create UnifiedDataset from raw data
  - Encode each sample with all models
  - Save tensors + JSON index to cache directory

**inference.py:**
- Location: `inference.py` lines 8-47
- Triggers: `python inference.py`
- Responsibilities:
  - Load pretrained WanVideoPipeline with LoRA weights
  - Load control video, reference image, and prompt
  - Generate video via conditional diffusion sampling
  - Save output video with configured FPS

## Error Handling

**Strategy:** Try-catch with fallback and logging

**Patterns:**
- Exception handling in preprocessing loop with per-sample error capture (`preprocess_cache.py` lines 192-204)
- Gradient checkpointing auto-enable with warning if disabled (`train_full_pipeline.py` lines 29-31)
- Path resolution fallback for cached data (`train_lora_cached.py` lines 49-51)
- Model device/dtype conversion with validation

## Cross-Cutting Concerns

**Logging:** 
- Print-based progress tracking via tqdm in training loops (`train_lora_cached.py` lines 228, 368)
- Loss metrics logged per batch and per epoch
- Checkpoint save confirmations

**Validation:**
- Tensor shape assertions in model forward passes
- Metadata validation in dataset loading (height, width, num_frames)
- Device type checks (CPU vs CUDA) before tensor operations

**Authentication:**
- Hugging Face hub tokens for model downloads (handled by `modelscope` + `huggingface_hub`)
- Environment setup via `export PYTHONPATH`

**Distributed Execution:**
- Accelerate framework for multi-GPU/multi-node training
- Model wrapping/unwrapping for distributed data parallel
- Main process checks for checkpoint saving (`train_lora_cached.py` lines 373-376)

---

*Architecture analysis: 2026-04-24*
