#!/bin/bash

# Standalone LoRA training script for Wan2.1-Fun-V1.1-1.3B-Control
#
# Before running:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Prepare your dataset and update paths below
# 3. Download model weights (will auto-download from ModelScope if not present)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add the standalone folder to PYTHONPATH so diffsynth can be imported
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# === CONFIGURATION - Update these paths for your setup ===
DATASET_BASE_PATH="${SCRIPT_DIR}/data/kling"
DATASET_METADATA_PATH="${SCRIPT_DIR}/data/kling/metadata_depth_pose_control.csv"
OUTPUT_PATH="${SCRIPT_DIR}/output/Wan2.1-Fun-V1.1-1.3B-Control_lora"

# === Training parameters ===
HEIGHT=512
WIDTH=512
DATASET_REPEAT=2
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32

# === GPU configuration ===
NUM_PROCESSES=1
NUM_MACHINES=1

accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --num_machines ${NUM_MACHINES} \
  --mixed_precision no \
  --dynamo_backend no \
  "${SCRIPT_DIR}/train_full_pipeline.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,control_video,reference_image" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "control_video,reference_image"
