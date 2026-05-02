#!/bin/bash
# ============================================================
# Stage 2 Walk-Forward — SLURM Job Array (C-Vol + A-Vol)
# Warm-starts from Stage 1 checkpoints, trains vol surface.
#
# REQUIRES: Stage 1 B0-B12 complete + smoke test passed.
#           Edit STAGE1_DIR / STAGE1_MU below to pick warm start.
#           Currently configured for B10 (modified hybrid, μ=0.75,
#           warmup 5000) — the best Stage 1 config on pooled RMSE.
#
# Submit with:
#   sbatch --array=0-1 slurm/run_stage2.sh
#
# Task → vol_type mapping:
#   0: cvol  (multiplicative, NSM-inspired — primary contribution)
#   1: avol  (direct softplus, standard baseline comparator)
# ============================================================

#SBATCH --job-name=pinn_stage2
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/s2_%A_%a.out
#SBATCH --error=slurm/logs/s2_%A_%a.err

# ── EDIT THESE after Stage 1 analysis ──────────────────────
# STAGE1_DIR: which Stage 1 run to warm-start from (dir name under runs/walk_forward)
# STAGE1_MU:  must match the μ that produced that checkpoint (used for arch instantiation)
STAGE1_DIR=modified_hybrid_mu0.75_warmup5000   # B10: best on pooled RMSE
STAGE1_MU=0.75

# ── Environment ────────────────────────────────────────────
cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stage 1 dir: $STAGE1_DIR"
echo "Stage 1 μ:   $STAGE1_MU"
echo "Date:        $(date)"
echo "============================================"

# ── Config lookup ──────────────────────────────────────────
VOL_TYPES=(cvol avol)
VOL_TYPE=${VOL_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Config: vol_type=$VOL_TYPE  warm-start=$STAGE1_DIR"
echo "============================================"

# ── Run ────────────────────────────────────────────────────
# Single training run per fold (val-best snapshot reporting).
# Warm-start pricing net is loaded from the Stage 1 checkpoint
# (which under val-best scheme also doesn't see the val window —
# so val is genuinely held-out for Stage 2).
python run_stage2.py \
    --vol_type $VOL_TYPE \
    --checkpoint_dir runs/walk_forward/$STAGE1_DIR \
    --rwf_mu $STAGE1_MU \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 10000 \
    --val_months 1 \
    --seed 42 \
    --output_dir runs/stage2

echo "============================================"
echo "Done: $(date)"
echo "============================================"
