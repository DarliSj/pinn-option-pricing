#!/bin/bash
# ============================================================
# Stage 2 smoke test: one fold, 1000 epochs, C-Vol only.
# Validates both B10 and B12 warm-starts before the full array.
#
# REQUIRES: Stage 1 B0-B12 must have finished.
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

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"

echo "=== Smoke test 1/2: B10 (μ=0.75) warm-start ==="
python run_stage2.py \
    --vol_type cvol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75_warmup5000 \
    --rwf_mu 0.75 \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 1000 \
    --val_months 1 \
    --folds Nov2020 \
    --output_dir runs/stage2_smoke_B10 \
    --no_plots

echo ""
echo "=== Smoke test 2/2: B12 (μ=0.25) warm-start ==="
python run_stage2.py \
    --vol_type cvol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.25_warmup5000 \
    --rwf_mu 0.25 \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 1000 \
    --val_months 1 \
    --folds Nov2020 \
    --output_dir runs/stage2_smoke_B12 \
    --no_plots

echo "Stage 2 smoke done: $(date)"
