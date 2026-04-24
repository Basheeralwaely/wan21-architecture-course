#!/bin/bash

# Dataset Preparation Script using FLAME + WiLor + MediaPipe body pose
# Generates combined face+hand+body mask videos

# Configuration
SIGNER="test"
# Fixed prompt for all videos (customize this for your signer)
PROMPT="Adult woman signing, black turtleneck, purple background"

INPUT_DIR="./data/${SIGNER}/videos"
OUTPUT_DIR="./data/${SIGNER}"
MODEL_DIR="./dataset_collection/models"

# Output resolution (default: 512x512)
TARGET_WIDTH=512
TARGET_HEIGHT=512

# Random seed for reproducible reference frame selection
SEED=42

# Set to true to re-generate mask videos; false to only generate CSV from existing masks
PROCESS_VIDEOS=true

# Build command
CMD="python dataset_collection/prepare_dataset.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_dir ${MODEL_DIR} \
    --seed ${SEED} \
    --target_width ${TARGET_WIDTH} \
    --target_height ${TARGET_HEIGHT} \
    --prompt \"${PROMPT}\" \
    --generate_csv"

if [ "${PROCESS_VIDEOS}" = true ]; then
    CMD="${CMD} --process_videos"
fi

echo "=========================================="
echo "Dataset Preparation (FLAME + WiLor masks)"
echo "=========================================="
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "Process videos: ${PROCESS_VIDEOS}"
echo "Prompt: ${PROMPT}"
echo ""

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "${INPUT_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | wc -l)
echo "Found ${VIDEO_COUNT} videos"
echo ""

# Run the script
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Processing complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Mask videos:      ${OUTPUT_DIR}/masks/"
echo "  - Reference images: ${OUTPUT_DIR}/reference/"
echo "  - Metadata CSV:     ${OUTPUT_DIR}/metadata.csv"
