#!/bin/bash
# ============================================================
# Stage 2 Walk-Forward — SLURM Job Array (C-Vol + A-Vol)
# Warm-starts from Stage 1 checkpoints, trains vol surface.
#
# REQUIRES: Stage 1 B2-B7 complete + smoke test passed.
#           Edit STAGE1_MU below to the μ you selected.
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

# ── EDIT THIS after Stage 1 analysis ───────────────────────
STAGE1_MU=0.75   # ← set to the μ you selected from Stage 1

# ── Environment ────────────────────────────────────────────
cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stage 1 μ:  $STAGE1_MU"
echo "Date:       $(date)"
echo "============================================"

# ── Config lookup ──────────────────────────────────────────
VOL_TYPES=(cvol avol)
VOL_TYPE=${VOL_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Config: vol_type=$VOL_TYPE  checkpoint=modified_hybrid_mu${STAGE1_MU}"
echo "============================================"

# ── Run ────────────────────────────────────────────────────
# Keeps plots on — vol surface + slices are core diagnostics
python run_stage2.py \
    --vol_type $VOL_TYPE \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu${STAGE1_MU} \
    --rwf_mu $STAGE1_MU \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 10000 \
    --seed 42 \
    --output_dir runs/stage2

echo "============================================"
echo "Done: $(date)"
echo "============================================"
