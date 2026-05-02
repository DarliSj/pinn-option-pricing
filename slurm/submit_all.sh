#!/bin/bash
# ============================================================
# Submit all benchmark jobs (B0-B12) — Stage 1 walk-forward.
# Run from project root: bash slurm/submit_all.sh
#
# Workflow:
#   1. Smoke test first (verifies GPU + env + the val-best
#      snapshot pipeline on one fold)
#   2. B0-B12 array only starts after smoke test exits cleanly
#
# After this finishes:
#   bash slurm/submit_stage2.sh      # warm-start Stage 2
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

mkdir -p slurm/logs

echo "=== Submitting smoke test ==="
SMOKE_ID=$(sbatch --parsable slurm/run_smoke_test.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting B0-B12 benchmark array (depends on smoke test) ==="
# --array=0-12 → 13 tasks (B0 through B12)
# --dependency=afterok: array only starts if smoke test exits cleanly
BENCH_ID=$(sbatch --parsable \
    --array=0-12 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_benchmark.sh)
echo "Benchmark array job ID: $BENCH_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo "Watch logs:    tail -f slurm/logs/slurm_${BENCH_ID}_0.out"
echo ""
echo "Task → config:"
echo "  Task  0 (${BENCH_ID}_0):  standard physics                   [B0]"
echo "  Task  1 (${BENCH_ID}_1):  standard hybrid                    [B1]"
echo "  Task  2 (${BENCH_ID}_2):  modified physics  μ=0.50           [B2]"
echo "  Task  3 (${BENCH_ID}_3):  modified physics  μ=0.75           [B3]"
echo "  Task  4 (${BENCH_ID}_4):  modified physics  μ=1.00           [B4]"
echo "  Task  5 (${BENCH_ID}_5):  modified hybrid   μ=0.50           [B5]"
echo "  Task  6 (${BENCH_ID}_6):  modified hybrid   μ=0.75           [B6]"
echo "  Task  7 (${BENCH_ID}_7):  modified hybrid   μ=1.00           [B7]"
echo "  Task  8 (${BENCH_ID}_8):  modified hybrid   μ=0.00           [B8]  RWF init fix"
echo "  Task  9 (${BENCH_ID}_9):  modified hybrid   μ=0.75 fixλ_data [B9]  decouple data"
echo "  Task 10 (${BENCH_ID}_10): modified hybrid   μ=0.75 warmup    [B10] PDE warmup"
echo "  Task 11 (${BENCH_ID}_11): modified hybrid   μ=0.50 warmup    [B11] μ-down + warmup"
echo "  Task 12 (${BENCH_ID}_12): modified hybrid   μ=0.25 warmup    [B12] low-μ + warmup"
echo ""
echo "After this finishes, build the master table:"
echo "  python scripts/build_master_table.py"
