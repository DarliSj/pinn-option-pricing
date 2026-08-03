#!/bin/bash
# ============================================================
# R1 — Regime-conditioned pricing net (SLURM job array, 6 tasks)
#
# HYPOTHESIS: hybrid underperforms physics because the data term is
# ILL-POSED, not badly weighted. The net sees today's spot (via m=F/K)
# but NOT today's volatility level, so the same (m,tau) had different
# correct prices in March vs September and the data loss can only learn
# a regime-average. Adding nu = 1-day-lagged ATM IV as a third INPUT
# makes the regression identifiable.
#
# Bucket-model evidence (9-fold, mean-fold RMSE, scripts run locally):
#   BS(sigma_fix)                       6.95
#   BS + residual(m,tau)                6.72
#   BS + residual(m,tau,nu)             6.23   <- beats every PINN we have
#   (B10 hybrid 6.75 | B3 physics 7.05 | GAM 7.26)
# NOTE: nu is a CONDITIONING INPUT. Using it directly as the pricing
# sigma scored 8.61 (noisy daily estimate x vega) — do not do that.
#
# 2x3 design, all 9 folds, 15k epochs:
#   task 0: hybrid warmup5000 gradnorm            (B10 same-code control)
#   task 1: hybrid warmup5000 gradnorm   + nu     <- R1 test
#   task 2: hybrid warmup5000 relobralo           (W1 winner, 9 folds)
#   task 3: hybrid warmup5000 relobralo  + nu     <- R1 test (expected best)
#   task 4: physics                               (B3 same-code control)
#   task 5: physics                      + nu     (NULL control: physics is
#                                                  nu-independent, so this
#                                                  should match task 4 —
#                                                  isolates "data becomes
#                                                  identifiable" from "extra
#                                                  input capacity helps")
#
# Everything -> runs/v2/r1/ — its OWN subtree, deliberately:
#   * task 2's config name collides with the 3-fold W1 relobralo run in
#     runs/v2/walk_forward/; writing there would silently overwrite it.
#   * all 6 R1 tasks run 9 folds, so they stay mutually comparable in one
#     table instead of being mixed with W1's 3-fold arms.
# Dir names get a `_nuatm` suffix when regime conditioning is on.
#
# Submit:  bash slurm/submit_r1.sh          (smoke gates the array)
#      or: sbatch --array=0-5 slurm/run_r1_regime.sh
#
# ACCEPTANCE: task 3 (or 1) beats BOTH its no-nu twin AND the physics
# control on mean-fold RMSE. Task 5 ~= task 4 (null control holds).
# ============================================================

#SBATCH --job-name=pinn_r1_regime
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=slurm/logs/r1_%A_%a.out
#SBATCH --error=slurm/logs/r1_%A_%a.err

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

# R1 gets its own subtree (see header) — never the v1 tree, and never
# on top of the W1 arms in runs/v2/walk_forward/.
RUN_ROOT="${RUN_ROOT:-runs/v2/r1}"

MODES=(hybrid   hybrid   hybrid    hybrid    physics physics)
WARMUPS=(5000   5000     5000      5000      0       0)
BALANCERS=(gradnorm gradnorm relobralo relobralo gradnorm gradnorm)
REGIMES=(none    atm_iv_lag none    atm_iv_lag none  atm_iv_lag)

T=$SLURM_ARRAY_TASK_ID
MODE=${MODES[$T]}; WU=${WARMUPS[$T]}; BAL=${BALANCERS[$T]}; REG=${REGIMES[$T]}

EXTRA=""
[ "$WU" -gt 0 ] && EXTRA="$EXTRA --data_loss_warmup $WU"
[ "$REG" != "none" ] && EXTRA="$EXTRA --regime_input $REG"

echo "============================================"
echo "Job ID:  $SLURM_JOB_ID   task $T"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Config:  mode=$MODE  mu=0.75  warmup=$WU  balancer=$BAL  regime=$REG"
echo "Output:  ${RUN_ROOT}/walk_forward"
echo "Date:    $(date)"
echo "============================================"

python run_walk_forward.py \
    --arch modified \
    --mode $MODE \
    --rwf_mu 0.75 \
    --epochs 15000 \
    --val_months 1 \
    --seed 42 \
    --balancer $BAL \
    --track_test_curve \
    --output_dir ${RUN_ROOT}/walk_forward \
    $EXTRA

echo "============================================"
echo "Done: $(date)"
echo "============================================"
