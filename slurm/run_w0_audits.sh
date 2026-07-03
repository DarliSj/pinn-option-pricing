#!/bin/bash
# ============================================================
# W0 analysis-only audits (no training) — one GPU node, both scripts.
#
# Runs the two diagnostic passes over EXISTING v1 checkpoints:
#   1. physics-fidelity audit (PINN vs analytic BS + arbitrage severity)
#   2. σ_θ-vs-market-IV overlay (is Stage 2 learning real vol?)
#
# These load checkpoints + evaluate on a dense grid (a few minutes each),
# so they must run on a compute node, NOT the login node. Outputs land in
# reports/diagnostic/ (CSVs merge across re-runs; PNGs overwrite).
#
# Submit:  sbatch slurm/run_w0_audits.sh
#   Override the fold/config subsets via env vars, e.g.:
#     FIDELITY_FOLDS=all sbatch slurm/run_w0_audits.sh
#     OVERLAY_FOLDS=Nov2020 sbatch slurm/run_w0_audits.sh
#
# Alternatively, for a quick interactive run instead of batch:
#   srun --partition=gpu-common --gres=gpu:2080:1 --mem=16G \
#        --cpus-per-task=4 --time=0:30:00 --pty bash
#   # then, inside the allocation:
#   conda activate pinn_env
#   python scripts/audit_physics_fidelity.py --folds Nov2020,Sep2020
#   python scripts/overlay_vol_surface.py
# ============================================================

#SBATCH --job-name=pinn_w0_audits
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:00
#SBATCH --output=slurm/logs/w0_%j.out
#SBATCH --error=slurm/logs/w0_%j.err

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

# Fold/config subsets (env-overridable; defaults are the data-rich folds).
FIDELITY_CONFIGS="${FIDELITY_CONFIGS:-standard_physics,modified_physics_mu0.5,modified_physics_mu0.75,modified_physics_mu1.0}"
FIDELITY_FOLDS="${FIDELITY_FOLDS:-Nov2020,Sep2020}"
OVERLAY_FOLDS="${OVERLAY_FOLDS:-Nov2020,Dec2020}"

echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date:    $(date)"
echo "============================================"

echo ""
echo "=== W0 audit 1/2: physics-fidelity (vs analytic BS + arb severity) ==="
python scripts/audit_physics_fidelity.py \
    --configs "$FIDELITY_CONFIGS" \
    --folds "$FIDELITY_FOLDS"

echo ""
echo "=== W0 audit 2/2: σ_θ vs market-IV overlay (Stage 2) ==="
python scripts/overlay_vol_surface.py \
    --folds "$OVERLAY_FOLDS"

echo ""
echo "============================================"
echo "W0 audits done: $(date)"
echo "Outputs: reports/diagnostic/{physics_fidelity.csv,vol_overlay_summary.csv,*.png}"
echo "============================================"
