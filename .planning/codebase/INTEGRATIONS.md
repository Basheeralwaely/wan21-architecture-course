# External Integrations

**Analysis Date:** 2026-04-24

## APIs & External Services

**Model Registries:**
- ModelScope (PAI) - Primary model source for Wan2.1-Fun-V1.1-1.3B-Control
  - SDK/Client: `modelscope` package (snapshot_download)
  - Usage: Downloads pretrained diffusion models from `modelscope.com` using model IDs like `PAI/Wan2.1-Fun-V1.1-1.3B-Control`
  - Used in: `diffsynth/core/loader/config.py` (line 65-71)

- Hugging Face Hub - Fallback/alternative model source
  - SDK/Client: `huggingface_hub` package (snapshot_download)
  - Usage: Downloads models from `huggingface.co` as alternative to ModelScope
  - Used in: `diffsynth/core/loader/config.py` (line 72-79)
  - Controlled by: Environment variable `DIFFSYNTH_DOWNLOAD_SOURCE` (defaults to "modelscope")

**Text Encoding Models:**
- Google UMT5-XXL (T5 Text Encoder)
  - Model ID: `Wan-AI/Wan2.1-T2V-1.3B` with origin pattern `google/umt5-xxl/`
  - Purpose: Encodes text prompts to embeddings for video generation
  - Loaded via: `ModelConfig` with model_id and origin_file_pattern
  - Used in: `diffsynth/pipelines/wan_video.py` (line 35)

- OpenCLIP XLM-RoBERTa Large ViT-Huge-14
  - Model ID: `PAI/Wan2.1-Fun-V1.1-1.3B-Control` with origin pattern `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
  - Purpose: Image CLIP feature extraction for reference images
  - Used in: `preprocess_cache.py` for reference image encoding
  - Type: Vision-language model

**Audio Processing:**
- Wav2Vec2 Large XLS-R-53 English
  - Origin pattern: `wav2vec2-large-xlsr-53-english/`
  - Purpose: Speech-to-video audio encoding
  - Loaded via: `Wav2Vec2Processor.from_pretrained()` in `diffsynth/pipelines/wan_video.py` (line 41, 169)
  - Optional: Used for audio-conditioned video generation

## Data Storage

**Databases:**
- Not applicable - Project is standalone training, no database backends

**File Storage:**
- Local filesystem only
  - Model cache: `./models/` (configurable via `DIFFSYNTH_MODEL_BASE_PATH`)
  - Training data: `./data/` with structured subdirectories per subject/signer
  - Preprocessed cache: `./cache/` with pre-computed latents and embeddings
  - Output LoRA weights: `./output/` with per-epoch safetensors files
  - Results: `./results/` with generated video outputs

- See `/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training/` directory structure:
  - `data/` - Input videos, control videos, reference images, metadata CSV files
  - `cache/` - Pre-cached VAE latents, CLIP features, T5 embeddings
  - `output/` - Trained LoRA weights in safetensors format
  - `results/` - Generated video outputs

**Caching Strategy:**
- Pre-computed encodings cached to disk during preprocessing phase
- Cached data includes:
  - VAE latents for input video, control video, and reference image
  - CLIP features for reference image
  - T5/UMT5 embeddings for text prompts
  - Metadata (height, width, num_frames)
- See `preprocess_cache.py` for caching implementation

## Authentication & Identity

**Auth Provider:**
- None required
- ModelScope and Hugging Face Hub downloads work without authentication by default
- Optional: Users can configure HuggingFace tokens for private models (not currently used)

**Implementation:**
- Model downloading handled transparently by `modelscope.snapshot_download()` and `huggingface_hub.snapshot_download()`
- No explicit API keys or credentials needed for public model access

## Model Source Configuration

**Download Source Selection:**
- Environment variable `DIFFSYNTH_DOWNLOAD_SOURCE` controls source
  - `"modelscope"` (default) - Downloads from ModelScope/PAI registry
  - `"huggingface"` - Downloads from Hugging Face Hub
  - Implementation: `diffsynth/core/loader/config.py` lines 39-46

**Model IDs Used:**
- `PAI/Wan2.1-Fun-V1.1-1.3B-Control` - Main diffusion model (DiT weights)
- `Wan-AI/Wan2.1-T2V-1.3B` - Text encoder model ID
- `Wan-AI/Wan2.2-S2V-14B` - Audio processor reference (optional, not currently loaded)

**Origin File Patterns (selective downloads):**
- `diffusion_pytorch_model*.safetensors` - DiT model weights
- `models_t5_umt5-xxl-enc-bf16.pth` - T5 text encoder
- `Wan2.1_VAE.pth` - Video VAE encoder/decoder
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` - CLIP image encoder
- Patterns allow selective download of specific model components

## Monitoring & Observability

**Error Tracking:**
- Not integrated - errors logged to console via Python exceptions

**Logs:**
- Console output only (stdout/stderr)
- Logging points:
  - Model loading progress: `print()` statements in `diffsynth/pipelines/wan_video.py`
  - Cache preprocessing: Progress bars via `tqdm` in `preprocess_cache.py`
  - Training loss: Logged during training in `train_lora_cached.py`
  - Inference timing: Timing measurements in `fast_inference.py` and `inference.py`
- No persistent logging to files

## CI/CD & Deployment

**Hosting:**
- On-premise/local - No cloud platform integration
- Training runs locally on user's GPU

**CI Pipeline:**
- None detected - No GitHub Actions, GitLab CI, or other CI/CD integration
- No automated testing or deployment

**Deployment Method:**
- Standalone Python scripts executed directly or via shell scripts
- Models downloaded automatically on first run
- LoRA weights saved as `.safetensors` files for inference

## Environment Configuration

**Required env vars:**
- `PYTHONPATH` - Set in shell scripts to include project root
- `DIFFSYNTH_DOWNLOAD_SOURCE` (optional) - "modelscope" (default) or "huggingface"
- `DIFFSYNTH_SKIP_DOWNLOAD` (optional) - "true" to skip model downloads if already cached
- `DIFFSYNTH_MODEL_BASE_PATH` (optional) - Base directory for model storage (defaults to `./models`)

**Optional env vars:**
- `TOKENIZERS_PARALLELISM` - Set to "false" to suppress tokenizer warnings
- `CUDA_VISIBLE_DEVICES` - Control GPU allocation (handled by accelerate)

**Secrets location:**
- No secrets management - No API keys, credentials, or auth tokens in codebase
- ModelScope and HuggingFace downloads work without authentication

## Data Flow

**Preprocessing Pipeline (step1_cache_data.sh):**
1. Load dataset metadata CSV from `data/{signer}/metadata_depth_pose_control.csv`
2. For each sample:
   - Load video frames from disk (Pillow + imageio)
   - Load control video (depth/pose guidance)
   - Load reference image
   - Download required models from ModelScope if needed:
     - DiT (diffusion transformer)
     - T5 text encoder
     - VAE encoder
     - CLIP image encoder
   - Encode prompt → T5 embeddings
   - Encode video frames → VAE latents
   - Encode control video → VAE latents
   - Encode reference image → VAE latents + CLIP features
   - Save all to `cache/{signer}/` as PyTorch tensors

**Training Pipeline (step2_train_lora.sh):**
1. Load pre-cached latents from disk
2. Load DiT model weights
3. Inject LoRA adapters into DiT (target modules: q,k,v,o,ffn.0,ffn.2)
4. Forward pass: latents → DiT with LoRA → predictions
5. Backward pass: compute loss, update LoRA weights
6. Save LoRA weights as safetensors every N epochs

**Inference Pipeline (fast_inference.py, inference.py):**
1. Load DiT model from ModelScope
2. Load pre-trained LoRA weights as safetensors
3. Load text encoder, VAE, CLIP encoder
4. Encode prompt → T5 embeddings
5. Load control video and reference image
6. Forward pass: noise → DiT (with LoRA) → video latents
7. Decode latents → video frames via VAE
8. Save video to disk via imageio-ffmpeg

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-04-24*
