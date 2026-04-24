#!/bin/bash

# Dataset Preparation Script for Wan2.1-Fun-V1.1-1.3B-Control
# This script processes your videos to create pose, face, depth, and reference data

# Configuration
signer="Olivia_depth"
INPUT_DIR="data/${signer}/videos"
OUTPUT_DIR="data/${signer}"

# Optional: Set target resolution (comment out to keep original)
TARGET_WIDTH=512
TARGET_HEIGHT=512

# Depth model options: DPT_Large (best quality), DPT_Hybrid (balanced), MiDaS_small (fastest)
DEPTH_MODEL="DPT_Hybrid"
POSE_METHOD="mediapipe"   # yolo, mediapipe
FACE_MODE="contours"   #"dense", "contours", "yolo68"

# Random seed for reproducible reference frame selection
SEED=42

# Fixed prompt for all videos (customize this for your signer)
PROMPT="Adult Woman signing, black turtleneck, purple background"

# Build command
CMD="python dataset_collection2/prepare_dataset.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --depth_model ${DEPTH_MODEL} \
    --pose_method ${POSE_METHOD} \
    --face_mode ${FACE_MODE} \
    --seed ${SEED} \
    --prompt \"${PROMPT}\" \
    --generate_csv"

# Add target size if specified
if [ -n "${TARGET_WIDTH}" ] && [ -n "${TARGET_HEIGHT}" ]; then
    CMD="${CMD} --target_width ${TARGET_WIDTH} --target_height ${TARGET_HEIGHT}"
fi

echo "=========================================="
echo "Dataset Preparation for Wan2.1-Fun Training"
echo "=========================================="
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Depth model: ${DEPTH_MODEL}"
echo "Prompt: ${PROMPT}"
echo ""

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "${INPUT_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | wc -l)
echo "Found ${VIDEO_COUNT} videos to process"
echo ""

# Run the script
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Combining depth + pose control videos..."
echo "=========================================="

# Depth+Pose combination mode: blend, channels, or hstack
DEPTH_POSE_MODE="blend"
BLEND_WEIGHT=0.5

python dataset_collection2/combine_depth_pose.py \
    --depth_dir "${OUTPUT_DIR}/depth" \
    --pose_dir "${OUTPUT_DIR}/pose" \
    --output_dir "${OUTPUT_DIR}/depth_pose" \
    --mode ${DEPTH_POSE_MODE} \
    --blend_weight ${BLEND_WEIGHT} \
    --generate_csv \
    --prompt "${PROMPT}"

echo ""
echo "=========================================="
echo "Processing complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Pose videos:      ${OUTPUT_DIR}/pose/"
echo "  - Face videos:      ${OUTPUT_DIR}/faces/"
echo "  - Depth videos:     ${OUTPUT_DIR}/depth/"
echo "  - Depth+Pose videos: ${OUTPUT_DIR}/depth_pose/"
echo "  - Reference images: ${OUTPUT_DIR}/reference/"
echo ""
echo "Metadata CSV files:"
echo "  - ${OUTPUT_DIR}/metadata_pose_control.csv"
echo "  - ${OUTPUT_DIR}/metadata_depth_control.csv"
echo "  - ${OUTPUT_DIR}/metadata_depth_pose_control.csv"
echo "  - ${OUTPUT_DIR}/metadata_face.csv"
