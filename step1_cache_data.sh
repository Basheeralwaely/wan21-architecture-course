#!/bin/bash

# Step 1: Cache Data
# ==================
# Run this ONCE to precompute VAE/text/CLIP encodings.
# This step is slow but only needs to run once per dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# === CONFIGURATION ===
SIGNER="Olivia_depth"
DATASET_BASE_PATH="${SCRIPT_DIR}/data/${SIGNER}"
DATASET_METADATA_PATH="${SCRIPT_DIR}/data/${SIGNER}/metadata_depth_pose_control.csv"
CACHE_OUTPUT_PATH="${SCRIPT_DIR}/cache/${SIGNER}"

HEIGHT=512
WIDTH=512

python "${SCRIPT_DIR}/preprocess_cache.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,control_video,reference_image" \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --output_path "${CACHE_OUTPUT_PATH}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --extra_inputs "control_video,reference_image" \
  --device "cuda"

echo ""
echo "Caching complete! Next run: ./step2_train_lora.sh"
