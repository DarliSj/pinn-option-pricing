#!/bin/bash
# ============================================================
# PINN Walk-Forward Benchmark — SLURM Job Array
# Runs B2–B7: modified MLP + RWF, all mu × mode combos
#
# Submit with:
#   sbatch --array=0-5 slurm/run_benchmark.sh
#
# Or a single config for testing:
#   sbatch --array=2 slurm/run_benchmark.sh
#
# Task → config mapping:
#   0: modified physics mu=0.50   (B2)
#   1: modified physics mu=0.75   (B3)
#   2: modified physics mu=1.00   (B4)
#   3: modified hybrid  mu=0.50   (B5)
#   4: modified hybrid  mu=0.75   (B6)
#   5: modified hybrid  mu=1.00   (B7)
# ============================================================

#SBATCH --job-name=pinn_benchmark
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/slurm_%A_%a.out
#SBATCH --error=slurm/logs/slurm_%A_%a.err

# ── Environment ────────────────────────────────────────────
cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date:       $(date)"
echo "============================================"

# ── Config lookup (task ID → mode + mu) ────────────────────
MODES=(physics physics physics hybrid hybrid hybrid)
MUS=(0.5 0.75 1.0 0.5 0.75 1.0)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
MU=${MUS[$SLURM_ARRAY_TASK_ID]}

echo "Config: arch=modified  mode=$MODE  mu=$MU"
echo "============================================"

# ── Run ────────────────────────────────────────────────────
python run_walk_forward.py \
    --arch modified \
    --mode $MODE \
    --rwf_mu $MU \
    --epochs 15000 \
    --seed 42 \
    --output_dir runs/walk_forward

echo "============================================"
echo "Done: $(date)"
echo "============================================"
