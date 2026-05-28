#!/bin/bash
#SBATCH --job-name=synth_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module load cuda/11.7.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdp-tabdiff

nvidia-smi || echo "nvidia-smi not available"

MODEL="${1:-all}"

SYNTHCITY_MODELS=(ctgan tvae adsgan dpgan pategan)
SDV_MODELS=(sdv_ctgan sdv_tvae sdv_gaussian_copula sdv_copula_gan)
NOVEL_MODELS=(cdp_tabdiff)
ALL_MODELS=("${SYNTHCITY_MODELS[@]}" "${SDV_MODELS[@]}" "${NOVEL_MODELS[@]}")

run_model() {
    echo "=== Running benchmark_${1}.py ==="
    python "src/benchmarking/benchmark_${1}.py"
}

is_known_model() {
    local needle="$1"
    for m in "${ALL_MODELS[@]}"; do
        [[ "$m" == "$needle" ]] && return 0
    done
    return 1
}

case "${MODEL}" in
    all)
        for m in "${ALL_MODELS[@]}"; do
            run_model "${m}" || echo "WARNING: ${m} failed, continuing..."
        done
        ;;
    evaluate)
        python src/benchmarking/evaluate_all.py
        ;;
    *)
        if is_known_model "${MODEL}"; then
            run_model "${MODEL}"
        else
            echo "Usage: sbatch submit_benchmark.sh [<model>|all|evaluate]"
            echo "Models: ${ALL_MODELS[*]}"
            exit 1
        fi
        ;;
esac

if [[ "${MODEL}" == "all" ]]; then
    echo "=== Running unified evaluation ==="
    python src/benchmarking/evaluate_all.py
fi
