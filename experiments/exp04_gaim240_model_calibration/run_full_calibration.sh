#!/bin/bash
# ==============================================================================
# Full-Resolution GAIM-240 Model Evaluation & Calibration Pipeline
#
# Runs 3 models on all 216 video pairs (full 720p native resolution, full video duration):
# 1. Baseline ColorVideoVDP (Isotropic Laplacian Weber Pyramid)
# 2. Oriented ColorVideoVDP (Steerable Weber Pyramid, K=4 orientations)
# 3. V1-MT ColorVideoVDP (Bio-inspired V1/MT spatiotemporal stage)
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/../.."

HEIGHT=720
WIDTH=1280
FPS=240.0
WORKERS=8
CHUNK_SIZE=60

echo "================================================================================"
echo "Starting GAIM-240 Full-Resolution Quality Predictions Extraction"
echo "Resolution : ${HEIGHT}x${WIDTH} @ ${FPS} fps (Full Video Duration)"
echo "Working Dir: $(pwd)"
echo "================================================================================"

# 1. Baseline ColorVideoVDP
echo -e "\n[1/3] Extracting Baseline ColorVideoVDP predictions..."
uv run python "$DIR/extract_model_predictions.py" \
    --model baseline \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --chunk_size $CHUNK_SIZE \
    --num_workers $WORKERS \
    --output_csv "$DIR/predictions_baseline_720p_full.csv"

# 2. Oriented ColorVideoVDP (Steerable)
echo -e "\n[2/3] Extracting Oriented ColorVideoVDP predictions..."
uv run python "$DIR/extract_model_predictions.py" \
    --model oriented \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --chunk_size $CHUNK_SIZE \
    --num_workers $WORKERS \
    --output_csv "$DIR/predictions_oriented_720p_full.csv"

# 3. V1-MT ColorVideoVDP
echo -e "\n[3/3] Extracting V1-MT ColorVideoVDP predictions..."
uv run python "$DIR/extract_model_predictions.py" \
    --model v1mt \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --chunk_size $CHUNK_SIZE \
    --num_workers $WORKERS \
    --output_csv "$DIR/predictions_v1mt_720p_full.csv"

# 4. Calibrate and Compare Models
echo -e "\n[4/4] Fitting Calibration Curves & Evaluating Generalized Performance..."
uv run python "$DIR/calibrate_and_evaluate.py" \
    --csv_files \
        "$DIR/predictions_baseline_720p_full.csv" \
        "$DIR/predictions_oriented_720p_full.csv" \
        "$DIR/predictions_v1mt_720p_full.csv" \
    --output_dir "$DIR"

echo "================================================================================"
echo "Full Calibration & Evaluation Pipeline Completed Successfully!"
echo "Results and plots saved to: $DIR"
echo "================================================================================"
