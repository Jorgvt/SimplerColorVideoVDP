#!/bin/bash
#SBATCH -J gaim240_dl_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=stdout/%x_%j.out
#SBATCH --error=stdout/%x_%j.err

echo "=========================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Assigned Node: $SLURM_JOB_NODELIST"
echo "CPUs allocated:$SLURM_CPUS_PER_TASK"
echo "Start Time:    $(date)"
echo "=========================================================="

mkdir -p stdout
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Executing GAIM-240 PyTorch Streaming DataLoader Benchmark via uv..."
uv run python experiments/exp03_gaim240_dataloader_benchmark/benchmark_dataloader.py \
    --csv_path /media/disk/vista/BBDD_video_image/GAIM240/jod.csv \
    --data_dir /media/disk/vista/BBDD_video_image/GAIM240 \
    --mode train \
    --num_frames 60 \
    --height 360 \
    --width 640 \
    --batch_size 1 \
    --workers_list 0 2 4 8 \
    --prefetch_factor 2 \
    --output_json benchmark_results.json

EXIT_CODE=$?
echo "=========================================================="
echo "Job completed with exit code $EXIT_CODE at $(date)"
echo "=========================================================="
exit $EXIT_CODE
