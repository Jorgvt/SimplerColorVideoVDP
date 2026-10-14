#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=calib_form
#SBATCH --chdir=/home/vitojor/SimplerColorVideoVDP/experiments/exp05_oriented_form_calibration
#SBATCH --output=/home/vitojor/SimplerColorVideoVDP/experiments/exp05_oriented_form_calibration/log_calibrate_form.log

# ==============================================================================
# Experiment 05: Calibration of Oriented Form Model on GAIM-240
#
# 1. Extracts spatial orientation energy tensors across 216 video pairs (720p @ 240fps)
# 2. Performs orientation pooling parameter sweep (beta_ori in [0.25, 6.0])
# 3. Fits calibration mapping on training split (168 pairs)
# 4. Evaluates generalization on holdout validation scenes (48 pairs)
# ==============================================================================

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/vitojor/SimplerColorVideoVDP/experiments/exp05_oriented_form_calibration

echo "================================================================================"
echo "Starting Oriented Form Model Calibration Pipeline"
echo "Working Dir: $(pwd)"
echo "Date       : $(date)"
echo "================================================================================"

uv run python calibrate_oriented_form.py \
    --device cuda \
    --output_dir .

echo -e "\n================================================================================"
echo "Oriented Form Calibration Pipeline Completed at $(date)"
echo "================================================================================"
