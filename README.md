# Standalone LoRA Training for Wan2.1-Fun-V1.1-1.3B-Control

This is a minimal standalone folder for training LoRA models on Wan2.1-Fun-V1.1-1.3B-Control.

## Two Training Modes

### Option A: Full Pipeline (Original)
Uses `train_full_pipeline.py` - encodes data on-the-fly each epoch.
- Simple single-step workflow
- Slower per epoch (re-encodes video/text/images every time)

### Option B: Cached Training (Recommended)
Uses `preprocess_cache.py` + `train_lora_cached.py` - pre-computes encodings once.
- **Much faster training** - no VAE/text encoding during training
- Easy to experiment with different hyperparameters
- Cached data can be reused across multiple training runs

## Quick Start (Cached Training)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset in data/ folder

# 3. Cache data (run once, takes time)
./step1_cache_data.sh

# 4. Train LoRA (fast, run multiple times to experiment)
./step2_train_lora.sh
```

## What Gets Cached?

| Data | Model Used | Time per Sample |
|------|------------|-----------------|
| Video → Latents | VAE Encoder | ~2-5s |
| Control Video → Latents | VAE Encoder | ~2-5s |
| Reference Image → Latents | VAE Encoder | ~0.5s |
| Reference Image → Features | CLIP Encoder | ~0.2s |
| Prompt → Embeddings | Text Encoder | ~0.5s |

**During cached training, only the DiT model runs** - everything else is loaded from disk.

## Files Structure

```
stand_alone_Wan_lora_training/
├── train_full_pipeline.py    # Original full training (slower)
├── preprocess_cache.py       # Step 1: Cache encodings
├── train_lora_cached.py      # Step 2: Train with cache (fast)
├── step1_cache_data.sh       # Shell script for caching
├── step2_train_lora.sh       # Shell script for training
├── train_lora.sh             # Original single-step script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Your dataset goes here
├── cache/                    # Cached encodings (after step 1)
├── output/                   # Training outputs
└── diffsynth/                # Minimal diffsynth package
```

## Configuration

### step1_cache_data.sh
```bash
DATASET_BASE_PATH="./data/kling"           # Your dataset folder
DATASET_METADATA_PATH="./data/.../meta.csv" # CSV with video paths
CACHE_OUTPUT_PATH="./cache"                 # Where to save cache
HEIGHT=512                                  # Video height
WIDTH=512                                   # Video width
```

### step2_train_lora.sh
```bash
LEARNING_RATE=1e-4        # Try: 1e-5, 5e-5, 1e-4, 2e-4
NUM_EPOCHS=5              # More epochs = better fit (watch for overfitting)
BATCH_SIZE=1              # Increase if you have VRAM
DATASET_REPEAT=50         # How many times to repeat dataset per epoch
LORA_RANK=32              # Higher = more capacity, more VRAM
```

## Dataset Format

Your metadata CSV should contain columns:
- `video` - Path to training video
- `control_video` - Path to control signal video (depth/pose)
- `reference_image` - Path to reference image
- `prompt` - Text description

## Memory Usage Comparison

| Mode | VRAM During Training |
|------|---------------------|
| Full Pipeline | ~16-24 GB (VAE + Text Encoder + CLIP + DiT) |
| Cached Training | ~8-12 GB (DiT only) |

## Tips

1. **Run caching overnight** - It's slow but only needs to happen once
2. **Experiment freely** - Cached training is fast, try different learning rates
3. **Watch the loss** - If it stops decreasing, training might be done
4. **Check outputs** - Run inference with your LoRA to validate quality

## Original Codebase

https://github.com/modelscope/DiffSynth-Studio
