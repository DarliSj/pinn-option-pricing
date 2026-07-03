#!/bin/bash
# ============================================================
# Submit W1 (balancer falsification): smoke → 6-task array.
# Mirrors submit_all.sh / submit_stage2.sh: the array only starts
# if the smoke test exits cleanly.
#
# All output → runs/v2/ (never the frozen v1 tree). See
# EXPERIMENTS.md and docs/TODO.md W1.
#
# Run from project root: bash slurm/submit_w1.sh
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

mkdir -p slurm/logs

echo "=== Submitting W1 smoke test (relobralo / gradnorm_renorm / fixed) ==="
SMOKE_ID=$(sbatch --parsable slurm/run_w1_smoke.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting W1 balancer array (6 tasks, depends on smoke) ==="
# --array=0-5 → {no-warmup, warmup5000} × {relobralo, gradnorm_renorm, fixed}
W1_ID=$(sbatch --parsable \
    --array=0-5 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_w1_balancer.sh)
echo "W1 array job ID: $W1_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo "Watch logs:    tail -f slurm/logs/w1_${W1_ID}_0.out"
echo ""
echo "Task → config (all: modified hybrid μ=0.75, folds Aug/Oct/Nov, → runs/v2):"
echo "  Task 0 (${W1_ID}_0): no-warmup   × relobralo"
echo "  Task 1 (${W1_ID}_1): no-warmup   × gradnorm_renorm"
echo "  Task 2 (${W1_ID}_2): no-warmup   × fixed"
echo "  Task 3 (${W1_ID}_3): warmup=5000 × relobralo"
echo "  Task 4 (${W1_ID}_4): warmup=5000 × gradnorm_renorm"
echo "  Task 5 (${W1_ID}_5): warmup=5000 × fixed"
echo ""
echo "Control = the existing v1 gradnorm runs of B6 (no-warmup) / B10 (warmup)."
echo "After it finishes, aggregate v2 and diff against v1:"
echo "  python scripts/build_master_table.py --runs_dir runs/v2 --output_dir reports/v2 --label v2"
echo "  python scripts/compare_methodologies.py"
