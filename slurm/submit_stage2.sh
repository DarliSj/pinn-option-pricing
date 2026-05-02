#!/bin/bash
# ============================================================
# Submit Stage 2: smoke test + 4-task array
#   (B10 × {cvol,avol}) + (B12 × {cvol,avol})
#
# REQUIRES:
#   1. Stage 1 B0-B12 array has finished successfully
#   2. Smoke test validates both B10 and B12 warm-starts
#
# Run from project root: bash slurm/submit_stage2.sh
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

echo "=== Submitting Stage 2 smoke test (B10 + B12) ==="
SMOKE_ID=$(sbatch --parsable slurm/run_stage2_smoke.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting Stage 2 array (4 tasks, depends on smoke) ==="
STAGE2_ID=$(sbatch --parsable \
    --array=0-3 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_stage2.sh)
echo "Stage 2 array job ID: $STAGE2_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo ""
echo "Task → config:"
echo "  Task 0 (${STAGE2_ID}_0): B10 (μ=0.75) × cvol  → runs/stage2_B10/cvol"
echo "  Task 1 (${STAGE2_ID}_1): B10 (μ=0.75) × avol  → runs/stage2_B10/avol"
echo "  Task 2 (${STAGE2_ID}_2): B12 (μ=0.25) × cvol  → runs/stage2_B12/cvol"
echo "  Task 3 (${STAGE2_ID}_3): B12 (μ=0.25) × avol  → runs/stage2_B12/avol"
