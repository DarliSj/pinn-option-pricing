#!/bin/bash
# ============================================================
# W1 — Balancer falsification test (SLURM job array, 6 tasks)
#
# Question: does BOUNDING the loss balancer kill the λ_data
# runaway and stop hybrid degrading vs physics? (See
# docs/TRAINING_VALIDATION_DISCUSSION.md P1 and docs/TODO.md W1.)
#
# Grid: {B6-style, B10-style} × {relobralo, gradnorm_renorm, fixed}
#   B6-style  = modified hybrid μ=0.75, NO warmup (worst v1 drifter)
#   B10-style = modified hybrid μ=0.75, warmup=5000 (v1 Pareto member)
# v1 gradnorm runs of the SAME configs/folds are the control —
# no need to re-run them.
#
# Folds: Aug2020, Oct2020, Nov2020 (where B6's hybrid degradation
# was clearest; Oct also stresses the Sep-val mismatch).
# track_test_curve ON (diagnostic only — feeds the W3 protocol
# decision: val-best vs final-epoch vs oracle).
#
# Outputs land in runs/v2/ (NEVER the frozen v1 tree):
#   runs/v2/walk_forward/modified_hybrid_mu0.75_<balancer>
#   runs/v2/walk_forward/modified_hybrid_mu0.75_warmup5000_<balancer>
#
# Submit:  RUN_ROOT=runs/v2 sbatch --array=0-5 slurm/run_w1_balancer.sh
#   (RUN_ROOT defaults to runs/v2 here anyway — this script never
#    writes into the v1 tree.)
#
# Smoke first — use the orchestrator (smoke gates the array):
#   bash slurm/submit_w1.sh
# or a one-off smoke by hand (gitignored transient dir):
#   python run_walk_forward.py --mode hybrid --arch modified \
#       --rwf_mu 0.75 --balancer relobralo --folds Nov2020 \
#       --epochs 1500 --data_loss_warmup 200 \
#       --output_dir runs/v2/smoke --track_test_curve
#
# ACCEPTANCE (per docs/TODO.md W1): weight trajectories PLATEAU
# (no λ runaway), and hybrid test RMSE stops degrading vs the
# physics counterpart on the same folds.
# ============================================================

#SBATCH --job-name=pinn_w1_balancer
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:30:00
#SBATCH --output=slurm/logs/w1_%A_%a.out
#SBATCH --error=slurm/logs/w1_%A_%a.err

# ── Environment ────────────────────────────────────────────
cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

# ── Output routing (methodology versioning; see EXPERIMENTS.md) ──
# W1 is new-methodology work → defaults to the v2 tree.
RUN_ROOT="${RUN_ROOT:-runs/v2}"

# ── 6-task lookup: warmup × balancer ───────────────────────
WARMUPS=(0 0 0 5000 5000 5000)
BALANCERS=(relobralo gradnorm_renorm fixed relobralo gradnorm_renorm fixed)

TASK=$SLURM_ARRAY_TASK_ID
WU=${WARMUPS[$TASK]}
BAL=${BALANCERS[$TASK]}

EXTRA=""
[ "$WU" -gt 0 ] && EXTRA="--data_loss_warmup $WU"

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID   task $TASK"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Config:     modified hybrid μ=0.75  warmup=$WU  balancer=$BAL"
echo "Output:     ${RUN_ROOT}/walk_forward"
echo "Date:       $(date)"
echo "============================================"

python run_walk_forward.py \
    --arch modified \
    --mode hybrid \
    --rwf_mu 0.75 \
    --epochs 15000 \
    --val_months 1 \
    --seed 42 \
    --folds Aug2020,Oct2020,Nov2020 \
    --balancer $BAL \
    --track_test_curve \
    --output_dir ${RUN_ROOT}/walk_forward \
    $EXTRA

echo "============================================"
echo "Done: $(date)"
echo "============================================"
