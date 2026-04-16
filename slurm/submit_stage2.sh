#!/bin/bash
# ============================================================
# Submit Stage 2: smoke test + both vol_types (C-Vol, A-Vol)
#
# REQUIRES:
#   1. Stage 1 B2-B7 array has finished successfully
#   2. You have selected a μ from Stage 1 walk-forward results
#   3. You edited STAGE1_MU in run_stage2_smoke.sh AND run_stage2.sh
#
# Run from project root: bash slurm/submit_stage2.sh
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

echo "=== Submitting Stage 2 smoke test ==="
SMOKE_ID=$(sbatch --parsable slurm/run_stage2_smoke.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting Stage 2 array (C-Vol + A-Vol, depends on smoke) ==="
STAGE2_ID=$(sbatch --parsable \
    --array=0-1 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_stage2.sh)
echo "Stage 2 array job ID: $STAGE2_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo ""
echo "Task → vol_type:"
echo "  Task 0 (${STAGE2_ID}_0): cvol  [multiplicative, primary]"
echo "  Task 1 (${STAGE2_ID}_1): avol  [direct, comparator]"
