#!/bin/bash
# ============================================================
# Stage 2 Walk-Forward — SLURM Job Array (4 tasks)
# Warm-starts from Stage 1 checkpoints, trains vol surface.
#
# REQUIRES: Stage 1 B0-B12 complete + smoke test passed.
#
# Two warm-start configs × two vol parameterizations = 4 tasks:
#   0: B10 (μ=0.75) × cvol  — best RMSE warm-start, multiplicative
#   1: B10 (μ=0.75) × avol  — best RMSE warm-start, direct
#   2: B12 (μ=0.25) × cvol  — cleanest PDE surface, multiplicative
#   3: B12 (μ=0.25) × avol  — cleanest PDE surface, direct
#
# Submit with:
#   sbatch --array=0-3 slurm/run_stage2.sh
# ============================================================

#SBATCH --job-name=pinn_stage2
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/s2_%A_%a.out
#SBATCH --error=slurm/logs/s2_%A_%a.err

# ── Environment ────────────────────────────────────────────
cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

# ── 4-task config lookup ──────────────────────────────────
# Rows: [STAGE1_DIR, STAGE1_MU, VOL_TYPE, OUTPUT_TAG]
STAGE1_DIRS=(
    modified_hybrid_mu0.75_warmup5000   # task 0: B10
    modified_hybrid_mu0.75_warmup5000   # task 1: B10
    modified_hybrid_mu0.25_warmup5000   # task 2: B12
    modified_hybrid_mu0.25_warmup5000   # task 3: B12
)
STAGE1_MUS=(0.75 0.75 0.25 0.25)
VOL_TYPES=(cvol avol cvol avol)
OUTPUT_TAGS=(stage2_B10 stage2_B10 stage2_B12 stage2_B12)

TASK=$SLURM_ARRAY_TASK_ID
STAGE1_DIR=${STAGE1_DIRS[$TASK]}
STAGE1_MU=${STAGE1_MUS[$TASK]}
VOL_TYPE=${VOL_TYPES[$TASK]}
OUTPUT_TAG=${OUTPUT_TAGS[$TASK]}

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $TASK"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stage 1 dir: $STAGE1_DIR  (μ=$STAGE1_MU)"
echo "Vol type:    $VOL_TYPE"
echo "Output:      runs/$OUTPUT_TAG/$VOL_TYPE"
echo "Date:        $(date)"
echo "============================================"

# ── Run ────────────────────────────────────────────────────
python run_stage2.py \
    --vol_type $VOL_TYPE \
    --checkpoint_dir runs/walk_forward/$STAGE1_DIR \
    --rwf_mu $STAGE1_MU \
    --pricing_lr 1e-4 \
    --vol_lr 1e-3 \
    --epochs 10000 \
    --val_months 1 \
    --seed 42 \
    --output_dir runs/$OUTPUT_TAG

echo "============================================"
echo "Done: $(date)"
echo "============================================"
