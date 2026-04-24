# Codebase Structure

**Analysis Date:** 2026-04-24

## Directory Layout

```
stand_alone_Wan_lora_training/
├── training_pipelines/          # Entry point scripts (root level)
│   ├── train_full_pipeline.py   # Full pipeline: encode + train per epoch (slow)
│   ├── train_lora_cached.py     # Fast cached training (recommended)
│   ├── preprocess_cache.py      # Step 1: Pre-encode all data
│   └── inference.py             # Generate videos with trained LoRA
│
├── execution_scripts/           # Shell wrappers (root level)
│   ├── step1_cache_data.sh      # Bash wrapper for preprocess_cache.py
│   ├── step2_train_lora.sh      # Bash wrapper for train_lora_cached.py
│   ├── train_lora.sh            # Original single-step training
│   └── run_prepare_dataset.sh   # Dataset preparation script
│
├── diffsynth/                   # Core DiffSynth framework
│   ├── __init__.py              # Package initialization
│   ├── configs/                 # Configuration management
│   │   ├── model_configs.py     # ModelConfig class for loading models
│   │   └── vram_management_module_maps.py
│   │
│   ├── core/                    # Low-level utilities
│   │   ├── attention/           # Custom attention implementations
│   │   ├── data/                # Dataset and operator utilities
│   │   │   ├── unified_dataset.py   # Flexible dataset class
│   │   │   └── operators.py     # Composable data operators (LoadVideo, LoadImage, etc.)
│   │   ├── device/              # Device management (CPU/CUDA/NPU)
│   │   ├── gradient/            # Gradient checkpointing
│   │   ├── loader/              # Model loading from HF/ModelScope
│   │   └── vram/                # VRAM optimization
│   │
│   ├── models/                  # Neural network implementations
│   │   ├── wan_video_dit.py     # DiT (Diffusion Transformer) - main training target
│   │   ├── wan_video_vae.py     # Video VAE encoder/decoder
│   │   ├── wan_video_text_encoder.py  # T5/UMT5 text embedding
│   │   ├── wan_video_image_encoder.py # CLIP vision encoder
│   │   ├── wan_video_vace.py    # VACE model variant
│   │   ├── wan_video_motion_controller.py  # Motion control adapter
│   │   ├── wan_video_animate_adapter.py   # Animation adapter
│   │   ├── wan_video_mot.py     # MOT model variant
│   │   ├── wan_video_dit_s2v.py # Sound-to-Video DiT
│   │   ├── wan_video_camera_controller.py # Camera control
│   │   ├── wav2vec.py           # Audio encoder (S2V)
│   │   ├── model_loader.py      # ModelLoader class
│   │   └── general_modules.py   # Shared layer definitions
│   │
│   ├── diffusion/               # Training framework
│   │   ├── __init__.py          # Export training classes
│   │   ├── training_module.py   # Base DiffusionTrainingModule class
│   │   ├── flow_match.py        # Flow matching scheduler
│   │   ├── loss.py              # Loss computation (FlowMatchSFTLoss, DirectDistillLoss)
│   │   ├── runner.py            # Training loop launcher functions
│   │   ├── logger.py            # ModelLogger for checkpoint management
│   │   ├── parsers.py           # CLI argument parsing utilities
│   │   └── base_pipeline.py     # BasePipeline and PipelineUnit classes
│   │
│   ├── pipelines/               # High-level pipelines
│   │   └── wan_video.py         # WanVideoPipeline (main orchestrator)
│   │
│   └── utils/                   # Utility modules
│       ├── lora/                # LoRA adaptation utilities
│       ├── controlnet/          # ControlNet input handling
│       ├── data/                # Data processing utilities
│       ├── state_dict_converters/ # Model state dict conversion
│       └── xfuser/              # Sequence parallel utilities
│
├── data/                        # Input dataset directories (organized by subject)
│   ├── Olivia/
│   │   ├── videos/              # Training videos
│   │   ├── reference/           # Reference images
│   │   ├── masks/               # (Optional) pre-computed masks
│   │   └── metadata*.csv        # Metadata index files
│   ├── Olivia_depth/            # Depth-controlled version
│   │   ├── videos/
│   │   ├── depth/               # Depth maps for control
│   │   ├── depth_pose/          # Combined depth+pose control
│   │   ├── reference/
│   │   ├── pose/                # 2D pose control
│   │   └── metadata*.csv
│   ├── Theo/
│   ├── kling/
│   └── test/                    # Test dataset
│
├── cache/                       # Pre-computed encodings (generated by preprocess_cache.py)
│   ├── Olivia_depth/            # Organized by dataset
│   │   ├── cache_index.json     # Index mapping samples to tensor locations
│   │   ├── sample_000000/       # One sample per directory
│   │   │   ├── input_latents.pt     # VAE-encoded video [B,16,T,H,W]
│   │   │   ├── context.pt           # Text-encoded prompt [B,L,4096]
│   │   │   ├── control_latents.pt   # VAE-encoded control video (optional)
│   │   │   ├── reference_latents.pt # VAE-encoded reference image (optional)
│   │   │   ├── clip_feature.pt      # CLIP-encoded reference image (optional)
│   │   │   └── metadata.json        # Sample metadata (height, width, frames, prompt)
│   │   ├── sample_000001/
│   │   └── ...
│   └── fast_inference/          # Fast inference cache
│
├── output/                      # Training outputs (generated by train_lora_cached.py)
│   ├── Olivia_depth/
│   │   └── Wan2.1-Fun-V1.1-1.3B-Control_lora/
│   │       ├── lora_epoch_1.safetensors    # Checkpoint per epoch
│   │       ├── lora_epoch_2.safetensors
│   │       └── lora_final.safetensors      # Final trained weights
│   ├── Olivia/
│   ├── Heather/
│   └── SLURM/                   # (Optional) SLURM batch job outputs
│
├── results/                     # Generated video outputs (from inference.py)
│   ├── FLUCTUATION@VARIATION^FS.mp4  # Sample output
│   └── ...
│
├── dataset_collection/          # Data preparation utilities (separate project)
│   ├── prepare_dataset.py       # Mask generation from videos (face/hand/body)
│   ├── face.py                  # FLAME face fitting
│   ├── hand.py                  # Hand pose detection
│   ├── render.py                # Rendering utilities
│   └── models/
│       └── mediapipe_landmark_embedding/
│
├── dataset_collection2/         # Alternative dataset collection (depth+pose based)
│   ├── prepare_dataset.py       # Depth/pose extraction
│   ├── pose2d.py                # 2D pose estimation
│   ├── pose2d_utils.py          # Pose utilities
│   ├── flame.py                 # FLAME model interface
│   ├── generate_expression.py   # Expression generation
│   ├── combine_depth_pose.py    # Combine modalities
│   └── yolo/
│       ├── det/                 # YOLO detection models
│       └── pose2d/              # YOLO pose models
│
├── test_files/                  # Test utilities
│   └── test_pt_files.py         # Tensor file testing
│
├── .planning/                   # GSD planning directory (generated)
│   └── codebase/                # Architecture analysis documents
│       ├── ARCHITECTURE.md      # This architecture analysis
│       └── STRUCTURE.md         # This structure analysis
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and quick start
├── model_architecture.md        # Detailed DiT block structure
└── Wan2.1-Fun-V1.1-1.3B-Control.py  # (Legacy) model config file
```

## Directory Purposes

**Training Pipeline (root level):**
- Purpose: Main entry points for the two-stage training workflow
- Contains: Python scripts for preprocessing, training, and inference
- Key files: `preprocess_cache.py`, `train_lora_cached.py`, `train_full_pipeline.py`, `inference.py`

**diffsynth/:**
- Purpose: Complete DiffSynth framework - all models, pipelines, and utilities
- Contains: Modular architecture for diffusion-based video generation
- Key subdirectories: `models/` (neural networks), `diffusion/` (training), `pipelines/` (orchestration)

**data/:**
- Purpose: Raw input datasets organized by subject/character
- Contains: Videos, reference images, control signals, metadata CSV files
- Naming convention: Subdirectories per subject (Olivia, Theo, kling), with modality subdirectories (videos, reference, depth, pose)

**cache/:**
- Purpose: Preprocessed tensor cache to enable fast training
- Contains: PyTorch `.pt` files with pre-encoded latents and embeddings
- Organization: Structured as `cache/[dataset_name]/sample_XXXXXX/` with JSON index
- Generated by: `preprocess_cache.py`

**output/:**
- Purpose: LoRA checkpoint outputs from training
- Contains: SafeTensor `.safetensors` files with trained LoRA weights
- Organization: `output/[dataset_name]/Wan2.1-Fun-V1.1-1.3B-Control_lora/`
- Generated by: `train_lora_cached.py` or `train_full_pipeline.py`

**dataset_collection/ and dataset_collection2/:**
- Purpose: Auxiliary scripts for data preparation (face detection, mask generation, pose extraction)
- Contains: Detection models, rendering tools, data transformation utilities
- Used for: Creating control signals from raw video

## Key File Locations

**Entry Points:**
- `preprocess_cache.py`: Preprocessing stage - encodes all data once
- `train_lora_cached.py`: Training stage - fast training with cached data
- `train_full_pipeline.py`: Legacy training - full pipeline per epoch
- `inference.py`: Generation stage - creates videos with trained LoRA

**Configuration:**
- `requirements.txt`: Python package dependencies
- `step1_cache_data.sh`: Configuration wrapper for preprocessing
- `step2_train_lora.sh`: Configuration wrapper for training
- `.planning/codebase/`: Architecture documentation (this directory)

**Core Logic:**
- `diffsynth/pipelines/wan_video.py`: Main WanVideoPipeline orchestrator
- `diffsynth/models/wan_video_dit.py`: DiT model (training target)
- `diffsynth/core/data/unified_dataset.py`: Flexible dataset loader
- `diffsynth/diffusion/training_module.py`: Base training harness

**Data Handling:**
- `data/`: Raw input videos and control signals
- `cache/`: Pre-encoded tensors (persistent across training runs)
- `output/`: LoRA weights (training outputs)
- `results/`: Generated videos (inference outputs)

## Naming Conventions

**Files:**
- `train_*.py`: Training entry points
- `preprocess_*.py`: Data preprocessing
- `wan_video_*.py`: Wan video model components
- `*.safetensors`: LoRA checkpoint files (safe tensor format)
- `*.pt`: PyTorch tensor files (cache data)
- `*_lora`: LoRA directory naming convention

**Directories:**
- `data/[subject_name]/`: Subject-organized datasets
- `data/[subject_name]/[modality]/`: Modality subdirectories (videos, depth, pose, reference)
- `cache/[dataset_name]/sample_XXXXXX/`: Numbered sample directories
- `output/[subject_name]/Wan2.1-*/`: Model-specific output directories

**Variables (from shell scripts):**
- `SIGNER`: Subject name (Olivia_depth, Theo, etc.)
- `DATASET_BASE_PATH`: Root data directory
- `DATASET_METADATA_PATH`: CSV metadata file
- `CACHE_OUTPUT_PATH`: Where to save cached tensors
- `OUTPUT_PATH`: Where to save trained LoRA weights

## Where to Add New Code

**New Feature (e.g., new loss function):**
- Primary code: `diffsynth/diffusion/loss.py` (add new loss class inheriting from base)
- Configuration: Update `train_full_pipeline.py` or `train_lora_cached.py` to reference new loss
- Integration: Update `task_to_loss` mapping in training module

**New Model Component (e.g., new encoder):**
- Implementation: `diffsynth/models/wan_video_*.py` (new file following naming pattern)
- Registration: Add to `WanVideoPipeline.from_pretrained()` model loading
- Integration: Add unit class to `WanVideoPipeline.units` list

**New Data Modality (e.g., new control signal):**
- Loader: Add new operator to `diffsynth/core/data/operators.py`
- Dataset: Update `UnifiedDataset.default_video_operator()` or create new operator chain
- Integration: Add to metadata CSV columns and pass via `data_file_keys`

**Utility Functions:**
- Shared helpers: `diffsynth/core/` subdirectories (attention, gradient, device, etc.)
- Model utilities: `diffsynth/utils/` subdirectories (lora, controlnet, data, etc.)
- Training utilities: `diffsynth/diffusion/` (loss, scheduler, logger, etc.)

## Special Directories

**cache/:**
- Purpose: Persistent cache of pre-encoded data
- Generated: Yes (created by `preprocess_cache.py`)
- Committed: No (should be in `.gitignore` - large binary files)
- Rebuild: Delete and re-run `preprocess_cache.py` if data changes

**output/:**
- Purpose: Training checkpoints and final LoRA weights
- Generated: Yes (created by training scripts)
- Committed: No (training artifacts too large)
- Rebuild: Re-run training to generate new checkpoints

**results/:**
- Purpose: Generated video outputs from inference
- Generated: Yes (created by `inference.py`)
- Committed: No (video files too large)
- Cleanup: Safe to delete between inference runs

**.planning/codebase/:**
- Purpose: Architecture and structure documentation for GSD planning
- Generated: Yes (created by GSD mapping agents)
- Committed: Yes (documentation should be tracked)
- Update: Re-run mapping when architecture changes significantly

---

*Structure analysis: 2026-04-24*
