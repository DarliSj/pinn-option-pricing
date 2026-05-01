#!/bin/bash
# ============================================================
# PINN Walk-Forward Benchmark — SLURM Job Array
# Runs B0–B10: arch × mode × μ + 3 loss-balancing ablations.
#
# Submit with:
#   sbatch --array=0-10 slurm/run_benchmark.sh
#
# Or a single config for testing:
#   sbatch --array=8 slurm/run_benchmark.sh
#
# Task → config mapping:
#   0: standard physics                              (B0)  naive baseline
#   1: standard hybrid                               (B1)  naive + market data
#   2: modified physics  μ=0.50                      (B2)
#   3: modified physics  μ=0.75                      (B3)
#   4: modified physics  μ=1.00                      (B4)
#   5: modified hybrid   μ=0.50                      (B5)
#   6: modified hybrid   μ=0.75                      (B6)  ← old Stage 2 default
#   7: modified hybrid   μ=1.00                      (B7)
#   8: modified hybrid   μ=0.00                      (B8)  fix #1: kill RWF init blowup
#   9: modified hybrid   μ=0.75  λ_data=1000 fixed  (B9)  fix #2: decouple data from grad-norm
#                                                          (1000 ≈ scale of bc weight at convergence
#                                                          for μ=0.75; gives data enough signal to
#                                                          compete without runaway to 10⁵+)
#  10: modified hybrid   μ=0.75  warmup=5000        (B10) fix #5: pre-train PDE then add data
# ============================================================

#SBATCH --job-name=pinn_benchmark
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
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

# ── Config lookup (task ID → arch + mode + μ + extras) ────
ARCHS=(standard standard modified modified modified modified modified modified modified modified modified)
MODES=(physics  hybrid   physics  physics  physics  hybrid   hybrid   hybrid   hybrid   hybrid   hybrid)
MUS=(   1.0     1.0      0.5      0.75     1.0      0.5      0.75     1.0      0.0      0.75     0.75)
FIXED_DATA=("" "" "" "" "" "" "" "" "" "1000.0" "")
WARMUP=(0 0 0 0 0 0 0 0 0 0 5000)

ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
MU=${MUS[$SLURM_ARRAY_TASK_ID]}
FD=${FIXED_DATA[$SLURM_ARRAY_TASK_ID]}
WU=${WARMUP[$SLURM_ARRAY_TASK_ID]}

# Build extra-args string from ablation flags
EXTRA=""
[ -n "$FD" ] && EXTRA="$EXTRA --fixed_data_weight $FD"
[ "$WU" -gt 0 ] && EXTRA="$EXTRA --data_loss_warmup $WU"

echo "Config: arch=$ARCH  mode=$MODE  μ=$MU  extras=[$EXTRA]"
echo "============================================"

# ── Run ────────────────────────────────────────────────────
# Single training run per fold (val-best snapshot reporting).
# val_months=1 (default) → val = month immediately before test month.
python run_walk_forward.py \
    --arch $ARCH \
    --mode $MODE \
    --rwf_mu $MU \
    --epochs 15000 \
    --val_months 1 \
    --seed 42 \
    --output_dir runs/walk_forward \
    $EXTRA

echo "============================================"
echo "Done: $(date)"
echo "============================================"
