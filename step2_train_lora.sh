#!/bin/bash

# Step 2: Train LoRA (Fast!)
# ==========================
# Run this after step1_cache_data.sh
# This step is much faster because it uses pre-cached latents.
# You can run this multiple times with different hyperparameters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# === CONFIGURATION ===
SIGNER="Olivia_depth"
CACHE_PATH="${SCRIPT_DIR}/cache/${SIGNER}"
OUTPUT_PATH="${SCRIPT_DIR}/output/${SIGNER}/Wan2.1-Fun-V1.1-1.3B-Control_lora"

# Training parameters - feel free to experiment!
LEARNING_RATE=1e-4
NUM_EPOCHS=5
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
DATASET_REPEAT=50
LORA_RANK=32
SAVE_EVERY_N_EPOCHS=1

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_lora_cached.py" \
  --cache_path "${CACHE_PATH}" \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors" \
  --output_path "${OUTPUT_PATH}" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank ${LORA_RANK} \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_repeat ${DATASET_REPEAT} \
  --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
  --use_gradient_checkpointing
