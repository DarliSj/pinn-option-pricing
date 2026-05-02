#!/bin/bash
# ============================================================
# History capture: re-run B6 (failure mode) and B10 (warmup fix)
# on Nov2020 ONLY, with the patched run_walk_forward.py that now
# saves the per-epoch `history` dict inside each fold .pt.
#
# Outputs go to runs/walk_forward_history/ — kept SEPARATE from
# the canonical Stage 1 dirs in runs/walk_forward/ so the existing
# Stage 2 warm-start sources are untouched.
#
# Two-task array:
#   0: B6  (modified hybrid, μ=0.75, no warmup) — failure mode
#   1: B10 (modified hybrid, μ=0.75, warmup=5000) — warmup fix
#
# Submit with:
#   sbatch --array=0-1 slurm/run_history_capture.sh
# ============================================================

#SBATCH --job-name=pinn_history
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/hist_%A_%a.out
#SBATCH --error=slurm/logs/hist_%A_%a.err

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

WARMUPS=(0 5000)             # task 0: B6 (no warmup), task 1: B10
WU=${WARMUPS[$SLURM_ARRAY_TASK_ID]}

EXTRA=""
[ "$WU" -gt 0 ] && EXTRA="--data_loss_warmup $WU"
TAG_HUMAN=$([ "$WU" -gt 0 ] && echo "B10" || echo "B6")

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID  ($TAG_HUMAN, warmup=$WU)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "============================================"

python run_walk_forward.py \
    --arch modified \
    --mode hybrid \
    --rwf_mu 0.75 \
    --epochs 15000 \
    --val_months 1 \
    --seed 42 \
    --folds Nov2020 \
    --output_dir runs/walk_forward_history \
    $EXTRA

echo "============================================"
echo "Done: $(date)"
echo "============================================"
