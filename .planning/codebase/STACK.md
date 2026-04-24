# Technology Stack

**Analysis Date:** 2026-04-24

## Languages

**Primary:**
- Python 3.10+ - Full codebase including model training, inference, data preprocessing, and utilities

**Secondary:**
- Bash - Shell scripts for orchestrating workflow (`step1_cache_data.sh`, `step2_train_lora.sh`, `train_lora.sh`)

## Runtime

**Environment:**
- Python 3.10.18 (identified in environment)
- CUDA-compatible GPU (referenced throughout with `torch.bfloat16` and device management)

**Package Manager:**
- pip (Python package manager)
- Lockfile: `requirements.txt` present at `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/requirements.txt`

## Frameworks

**Core ML/Training:**
- PyTorch (`torch>=2.0.0`) - Deep learning framework for model training and inference
- Transformers (`transformers>=4.36.0`) - Hugging Face transformers for model loading and tokenizers
- Accelerate (`accelerate>=0.25.0`) - Distributed training and mixed precision support
- PEFT (`peft>=0.7.0`) - Parameter-Efficient Fine-Tuning (LoRA implementation)

**Model & Audio:**
- HuggingFace Hub (`huggingface_hub>=0.20.0`) - Model downloading from Hugging Face Hub
- ModelScope (`modelscope>=1.10.0`) - Model downloading from ModelScope registry
- Wav2Vec2Processor (from `transformers`) - Audio processing for S2V (speech-to-video)

**Data & Image Processing:**
- Pillow (`Pillow>=10.0.0`) - Image loading and manipulation
- NumPy (`numpy>=1.24.0`) - Numerical computing
- Pandas (`pandas>=2.0.0`) - Data manipulation and CSV handling
- imageio (`imageio>=2.31.0`) - Video I/O
- imageio-ffmpeg (`imageio-ffmpeg>=0.4.9`) - FFmpeg backend for video encoding/decoding
- torchvision (`torchvision>=0.15.0`) - Vision utilities for PyTorch

**Model Serialization:**
- SafeTensors (`safetensors>=0.4.0`) - Safe model weight serialization and loading

**Utilities:**
- einops (`einops>=0.7.0`) - Tensor reshaping operations
- tqdm (`tqdm>=4.66.0`) - Progress bars for training loops
- typing_extensions (`typing_extensions>=4.8.0`) - Extended type hints
- ftfy (`ftfy>=6.1.0`) - Unicode text fixing

**Optional (Commented):**
- flash-attn (`>=2.3.0`) - Fast attention implementation (commented out, not currently used)

## Build/Dev Tools

**Training Orchestration:**
- Accelerate - Handles distributed training, mixed precision (bfloat16), and multi-GPU setup
- See `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/step2_train_lora.sh` for `accelerate launch` configuration

**Environment Variables:**
- `TOKENIZERS_PARALLELISM=false` - Set in training scripts to prevent tokenizer warnings
- `DIFFSYNTH_DOWNLOAD_SOURCE` - Controls download source (modelscope or huggingface)
- `DIFFSYNTH_SKIP_DOWNLOAD` - Skip model downloads if already present
- `DIFFSYNTH_MODEL_BASE_PATH` - Base path for model storage (defaults to `./models`)
- `PYTHONPATH` - Set in shell scripts to include project root

## Key Dependencies

**Critical for Training:**
- **PyTorch** (`>=2.0.0`) - Entire training/inference pipeline depends on this
- **PEFT** (`>=0.7.0`) - LoRA fine-tuning is core to this project
- **Transformers** (`>=4.36.0`) - Text encoding (T5/UMT5), tokenization, and model loading

**Critical for Model Loading:**
- **HuggingFace Hub** (`>=0.20.0`) - Can download from huggingface.co
- **ModelScope** (`>=1.10.0`) - Downloads models from ModelScope (PAI registry)
- **SafeTensors** (`>=0.4.0`) - Safe weight serialization and loading

**Critical for Data Processing:**
- **Pillow** (`>=10.0.0`) - Image loading for reference images
- **imageio + imageio-ffmpeg** - Video loading and encoding
- **NumPy** (`>=1.24.0`) - Tensor operations
- **Pandas** (`>=2.0.0`) - Metadata CSV parsing

**Infrastructure:**
- **Accelerate** (`>=0.25.0`) - Mixed precision training (bfloat16), distributed training
- **tqdm** (`>=4.66.0`) - Progress visualization during preprocessing and training

## Configuration

**Environment:**
- Configuration happens via command-line arguments in Python scripts
- Shell scripts (`step1_cache_data.sh`, `step2_train_lora.sh`) export `PYTHONPATH` and pass arguments
- See sections for environment variable control points

**Build/Training Configuration:**
- See `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/step1_cache_data.sh`:
  - Dataset paths
  - Output cache paths
  - Image dimensions (HEIGHT=512, WIDTH=512)
  - Model IDs and origin file patterns
  
- See `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/step2_train_lora.sh`:
  - Learning rate (default: 1e-4)
  - Number of epochs (default: 5)
  - Batch size (default: 1)
  - LoRA rank (default: 32)
  - Accelerate mixed precision setup (bfloat16)
  - LoRA target modules: "q,k,v,o,ffn.0,ffn.2"

**Model Configuration:**
- Models loaded via `ModelConfig` class from `diffsynth/core/loader/config.py`
- Default model source: ModelScope registry (PAI/Wan2.1-Fun-V1.1-1.3B-Control)
- Supports both local paths and remote model IDs with file patterns

## Platform Requirements

**Development:**
- CUDA-compatible GPU (required for bfloat16 training)
- Linux or macOS (bash scripts used)
- Python 3.10+ with pip
- ~16-24 GB VRAM for full pipeline training
- ~8-12 GB VRAM for cached training

**Production:**
- Same as development (this is a training/inference codebase, not web service)
- Deployment target: Local/on-premise GPU clusters
- Model outputs saved as `.safetensors` files (LoRA weights)

## Dependency Tree

```
Training Pipeline:
├── PyTorch (>=2.0.0)
│   ├── PEFT (>=0.7.0) - LoRA
│   ├── Transformers (>=4.36.0) - Text encoding, tokenizers
│   │   └── HuggingFace Hub (>=0.20.0) or ModelScope (>=1.10.0)
│   ├── Accelerate (>=0.25.0) - Mixed precision, distributed training
│   ├── torchvision (>=0.15.0)
│   └── SafeTensors (>=0.4.0) - Model serialization
├── Data Processing
│   ├── Pillow (>=10.0.0) - Images
│   ├── imageio (>=2.31.0) + imageio-ffmpeg (>=0.4.9) - Video I/O
│   ├── NumPy (>=1.24.0)
│   ├── Pandas (>=2.0.0) - Metadata
│   └── einops (>=0.7.0) - Tensor ops
└── Utilities
    ├── tqdm (>=4.66.0) - Progress
    ├── typing_extensions (>=4.8.0)
    └── ftfy (>=6.1.0) - Text fixing
```

---

*Stack analysis: 2026-04-24*
