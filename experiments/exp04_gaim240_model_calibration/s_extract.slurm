#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=cvvdp
#SBATCH --output=log_extract.log

# ==============================================================================
# Full-Resolution GAIM-240 Model Evaluation & Calibration Pipeline
#
# Runs 3 models on all 216 video pairs (full 720p native resolution, full video duration):
# 1. Baseline ColorVideoVDP (Isotropic Laplacian Weber Pyramid)
# 2. Oriented ColorVideoVDP (Steerable Weber Pyramid, K=4 orientations)
# 3. V1-MT ColorVideoVDP (Bio-inspired V1/MT spatiotemporal stage)
# ==============================================================================

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HEIGHT=720
WIDTH=1280
FPS=240.0
WORKERS=16
CHUNK_SIZE=15

echo "================================================================================"
echo "Starting GAIM-240 Full-Resolution Quality Predictions Extraction"
echo "Resolution : ${HEIGHT}x${WIDTH} @ ${FPS} fps (Full Video Duration)"
echo "Working Dir: $(pwd)"
echo "================================================================================"

# 1. Baseline ColorVideoVDP
# echo -e "\n[1/3] Extracting Baseline ColorVideoVDP predictions..."
# uv run "extract_model_predictions.py" \
#     --model baseline \
#     --height $HEIGHT \
#     --width $WIDTH \
#     --fps $FPS \
#     --chunk_size $CHUNK_SIZE \
#     --num_workers $WORKERS \
#     --output_csv "predictions_baseline_720p_full.csv"

# 2. Oriented ColorVideoVDP (Steerable)
# echo -e "\n[2/3] Extracting Oriented ColorVideoVDP predictions..."
# uv run "extract_model_predictions.py" \
#     --model oriented \
#     --height $HEIGHT \
#     --width $WIDTH \
#     --fps $FPS \
#     --chunk_size $CHUNK_SIZE \
#     --num_workers $WORKERS \
#     --output_csv "predictions_oriented_720p_full.csv"

# 3. V1-MT ColorVideoVDP
echo -e "\n[3/3] Extracting V1-MT ColorVideoVDP predictions..."
uv run "extract_model_predictions.py" \
    --model v1mt \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --chunk_size $CHUNK_SIZE \
    --num_workers $WORKERS \
    --output_csv "predictions_v1mt_720p_full.csv"

echo "================================================================================"
echo "Full Calibration Done!"
echo "================================================================================"
