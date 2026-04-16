#!/bin/bash
# ============================================================
# Submit all benchmark jobs.
# Run from project root: bash slurm/submit_all.sh
#
# Workflow:
#   1. Smoke test first (verifies GPU + env)
#   2. B2-B7 array only starts after smoke test succeeds
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

mkdir -p slurm/logs

echo "=== Submitting smoke test ==="
SMOKE_ID=$(sbatch --parsable slurm/run_smoke_test.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting B2-B7 benchmark array (depends on smoke test) ==="
# --array=0-5 → 6 tasks (B2 through B7)
# --dependency=afterok: array only starts if smoke test exits cleanly
BENCH_ID=$(sbatch --parsable \
    --array=0-5 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_benchmark.sh)
echo "Benchmark array job ID: $BENCH_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo "Watch logs:    tail -f slurm/logs/slurm_${BENCH_ID}_0.out"
echo ""
echo "Task → config:"
echo "  Task 0 (${BENCH_ID}_0): modified physics mu=0.50  [B2]"
echo "  Task 1 (${BENCH_ID}_1): modified physics mu=0.75  [B3]"
echo "  Task 2 (${BENCH_ID}_2): modified physics mu=1.00  [B4]"
echo "  Task 3 (${BENCH_ID}_3): modified hybrid  mu=0.50  [B5]"
echo "  Task 4 (${BENCH_ID}_4): modified hybrid  mu=0.75  [B6]"
echo "  Task 5 (${BENCH_ID}_5): modified hybrid  mu=1.00  [B7]"
