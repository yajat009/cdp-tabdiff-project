#!/bin/bash
#SBATCH --job-name=cdp_ablations
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

module load cuda/11.7.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdp-tabdiff

nvidia-smi || echo "nvidia-smi not available"

# ----------------------------------------------------------------------------
# Usage:
#   sbatch submit_ablations.sh                  # all 5 variants + evaluate
#   sbatch submit_ablations.sh base_tabddpm     # single variant
#   sbatch submit_ablations.sh fair_tabdiff
#   sbatch submit_ablations.sh dp_tabddpm_dense
#   sbatch submit_ablations.sh dp_tabddpm_random
#   sbatch submit_ablations.sh cdp_tabdiff
#   sbatch submit_ablations.sh evaluate         # re-run evaluator only
# ----------------------------------------------------------------------------

VARIANT="${1:-all}"

# Training hyperparameters (mirror the production CDP-TabDiff run).
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMBED_DIM="${EMBED_DIM:-32}"
N_BLOCKS="${N_BLOCKS:-4}"
TIMESTEPS="${TIMESTEPS:-1000}"
SCHEDULE="${SCHEDULE:-cosine}"
EPSILON="${EPSILON:-8.0}"
DELTA="${DELTA:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR="${LR:-1e-3}"
STROKE_WEIGHT="${STROKE_WEIGHT:-5.0}"

ABLATION_VARIANTS=(base_tabddpm fair_tabdiff dp_tabddpm_dense dp_tabddpm_random cdp_tabdiff)

run_variant() {
    local v="$1"
    echo "=== Running ablation: ${v} ==="
    python src/benchmarking/run_ablations.py \
        --variant "${v}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --embed-dim "${EMBED_DIM}" \
        --n-blocks "${N_BLOCKS}" \
        --timesteps "${TIMESTEPS}" \
        --schedule "${SCHEDULE}" \
        --epsilon "${EPSILON}" \
        --delta "${DELTA}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --lr "${LR}" \
        --stroke-weight "${STROKE_WEIGHT}"
}

is_known_variant() {
    local needle="$1"
    for m in "${ABLATION_VARIANTS[@]}"; do
        [[ "$m" == "$needle" ]] && return 0
    done
    return 1
}

case "${VARIANT}" in
    all)
        for v in "${ABLATION_VARIANTS[@]}"; do
            run_variant "${v}" || echo "WARNING: ${v} failed, continuing..."
        done
        ;;
    evaluate)
        python src/benchmarking/evaluate_all.py
        ;;
    *)
        if is_known_variant "${VARIANT}"; then
            run_variant "${VARIANT}"
        else
            echo "Usage: sbatch submit_ablations.sh [<variant>|all|evaluate]"
            echo "Variants: ${ABLATION_VARIANTS[*]}"
            exit 1
        fi
        ;;
esac

if [[ "${VARIANT}" == "all" ]]; then
    echo "=== Running unified evaluation ==="
    python src/benchmarking/evaluate_all.py
fi
