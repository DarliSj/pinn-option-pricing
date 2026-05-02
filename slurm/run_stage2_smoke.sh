#!/bin/bash
# ============================================================
# Stage 2 smoke test: one fold, 1000 epochs, C-Vol only.
# Verifies the warm-start + vol_model pipeline works before
# submitting the full Stage 2 array.
#
# REQUIRES: Stage 1 B0-B12 must have finished.
#           Edit STAGE1_DIR / STAGE1_MU below to pick warm start.
#           Currently configured for B10 (best on pooled RMSE).
#
# Submit with:
#   sbatch slurm/run_stage2_smoke.sh
# ============================================================

#SBATCH --job-name=pinn_s2_smoke
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/s2_smoke_%j.out
#SBATCH --error=slurm/logs/s2_smoke_%j.err

# ── EDIT THESE after Stage 1 analysis ──────────────────────
STAGE1_DIR=modified_hybrid_mu0.75_warmup5000   # B10: best on pooled RMSE
STAGE1_MU=0.75

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stage 1 dir: $STAGE1_DIR"
echo "Stage 1 μ:   $STAGE1_MU"
echo "Date: $(date)"

python run_stage2.py \
    --vol_type cvol \
    --checkpoint_dir runs/walk_forward/$STAGE1_DIR \
    --rwf_mu $STAGE1_MU \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 1000 \
    --val_months 1 \
    --folds Nov2020 \
    --output_dir runs/stage2_smoke \
    --no_plots

echo "Stage 2 smoke done: $(date)"
