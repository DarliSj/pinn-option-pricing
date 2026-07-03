#!/bin/bash
# ============================================================
# W1 smoke test: exercises the NEW balancer code paths on one
# fold at short epochs BEFORE the 6-task array commits ~15 GPU-h.
#
# Covers, per balancer (relobralo / gradnorm_renorm / fixed):
#   - the balancer weight-update branch in src/training.py
#   - the warmup→hybrid transition (short warmup=200 so it fires early)
#   - the always-on gradient-alignment diagnostic (grad_cosine)
#   - track_test_curve recording
#   - severity + medAPE metrics in the saved results.json
# All output → runs/v2/smoke/ (gitignored transient; NEVER the v1 tree).
#
# Submit:  sbatch slurm/run_w1_smoke.sh
# (Usually launched indirectly by slurm/submit_w1.sh as the gate.)
# ============================================================

#SBATCH --job-name=pinn_w1_smoke
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/w1_smoke_%j.out
#SBATCH --error=slurm/logs/w1_smoke_%j.err

set -e

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 1500 epochs > grad_norm_freq(1000) so the balancer fires twice; warmup=200
# so the warmup→hybrid transition (and post-warmup data balancing) is tested.
for BAL in relobralo gradnorm_renorm fixed; do
    echo ""
    echo "=== W1 smoke: balancer=$BAL ==="
    python run_walk_forward.py \
        --arch modified \
        --mode hybrid \
        --rwf_mu 0.75 \
        --epochs 1500 \
        --data_loss_warmup 200 \
        --val_months 1 \
        --folds Nov2020 \
        --balancer $BAL \
        --track_test_curve \
        --output_dir runs/v2/smoke
done

echo ""
echo "W1 smoke done: $(date)"
