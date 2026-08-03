#!/bin/bash
# ============================================================
# Submit R1 (regime-conditioned pricing net): smoke -> 6-task array.
# Mirrors submit_all.sh / submit_w1.sh — the array only starts if the
# smoke exits cleanly.
#
# All output -> runs/v2/. See EXPERIMENTS.md and docs/TODO.md.
#
# Run from project root: bash slurm/submit_r1.sh
# ============================================================

set -e

PROJECT=/hpc/group/fisherlab/ds555/pinn_code
cd $PROJECT

mkdir -p slurm/logs

echo "=== Submitting R1 smoke (no-nu and +nu paths, GPU) ==="
SMOKE_ID=$(sbatch --parsable slurm/run_r1_smoke.sh)
echo "Smoke job ID: $SMOKE_ID"

echo ""
echo "=== Submitting R1 array (6 tasks, 9 folds each, depends on smoke) ==="
R1_ID=$(sbatch --parsable \
    --array=0-5 \
    --dependency=afterok:$SMOKE_ID \
    slurm/run_r1_regime.sh)
echo "R1 array job ID: $R1_ID"

echo ""
echo "=== Submitted ==="
echo "Monitor with:  squeue -u $USER"
echo "Watch logs:    tail -f slurm/logs/r1_${R1_ID}_3.out    # the headline task"
echo ""
echo "Task -> config (all modified, mu=0.75, 9 folds, 15k epochs):"
echo "  Task 0 (${R1_ID}_0): hybrid warmup gradnorm            [B10 same-code control]"
echo "  Task 1 (${R1_ID}_1): hybrid warmup gradnorm   + nu     <- R1 test"
echo "  Task 2 (${R1_ID}_2): hybrid warmup relobralo           [W1 winner, 9 folds]"
echo "  Task 3 (${R1_ID}_3): hybrid warmup relobralo  + nu     <- R1 test (expected best)"
echo "  Task 4 (${R1_ID}_4): physics                           [B3 same-code control]"
echo "  Task 5 (${R1_ID}_5): physics                  + nu     [NULL control ~= task 4]"
echo ""
echo "ACCEPTANCE: task 3 (or 1) beats BOTH its no-nu twin AND the physics"
echo "control on mean-fold RMSE; task 5 ~= task 4."
echo ""
echo "When it finishes (R1 lives in its own subtree so W1's arms stay intact):"
echo "  python scripts/build_master_table.py --runs_dir runs/v2/r1 --output_dir reports/v2_r1 --label r1"
echo "  python scripts/compare_methodologies.py \\"
echo "      --v1 reports/v1/master_table.csv --v2 reports/v2_r1/master_table.csv \\"
echo "      --v2_label r1 --out reports/comparison_v1_r1.csv"
