"""Train CDP-TabDiff and emit synthetic stroke samples for benchmarking.

Produces ``results/synthetic_stroke_cdp_tabdiff.csv`` so the unified
``evaluate_all.py`` picks it up alongside the SynthCity / SDV baselines.

Usage
-----
    python src/benchmarking/benchmark_cdp_tabdiff.py
    python src/benchmarking/benchmark_cdp_tabdiff.py --epochs 5 --epsilon 8
    python src/benchmarking/benchmark_cdp_tabdiff.py --epsilon-sensitivity

Run with ``--smoke`` for a fast end-to-end sanity check (tiny data, 2
epochs, lax DP budget) — useful for verifying the Opacus integration on
a new node without committing to a long training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from benchmarking.config import (
    CDP_TABDIFF_EPSILON,
    CDP_TABDIFF_EPSILON_SENSITIVITY,
    DATASET_NAME,
    RESULTS_DIR,
    STROKE_DIM_LOSS_WEIGHT,
    STROKE_SAMPLE_LOSS_WEIGHT,
    USE_OVERSAMPLING,
    USE_REBALANCING,
)
from benchmarking.diffusion_runner import DiffusionRunConfig, run_diffusion_benchmark


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--epsilon", type=float, default=CDP_TABDIFF_EPSILON)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stroke-dim-weight", type=float, default=STROKE_DIM_LOSS_WEIGHT)
    p.add_argument("--stroke-sample-weight", type=float, default=STROKE_SAMPLE_LOSS_WEIGHT)
    p.add_argument(
        "--no-oversampling",
        action="store_true",
        help="Disable RandomOverSampler on the training fold.",
    )
    p.add_argument(
        "--no-rebalancing",
        action="store_true",
        help="Disable post-hoc positive-rate rebalancing.",
    )
    p.add_argument(
        "--epsilon-sensitivity",
        action="store_true",
        help=(
            "Also run sensitivity experiments at "
            f"{CDP_TABDIFF_EPSILON_SENSITIVITY}."
        ),
    )
    p.add_argument(
        "--learned-dag",
        action="store_true",
        help="Discover the causal mask with PC (causal-learn) instead of the expert DAG.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end run (subsample, 2 epochs, lax DP budget).",
    )
    return p.parse_args()


def _run_one(
    args: argparse.Namespace,
    epsilon: float,
    suffix: str = "",
    sample_size: int | None = None,
) -> pd.DataFrame:
    name = f"cdp_tabdiff{suffix}"
    cfg = DiffusionRunConfig(
        model_name=name,
        mask_type="learned" if args.learned_dag else "causal",
        dp_enabled=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        timesteps=args.timesteps,
        schedule_kind=args.schedule,
        target_epsilon=epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        stroke_dim_loss_weight=args.stroke_dim_weight,
        stroke_sample_loss_weight=args.stroke_sample_weight,
        use_oversampling=not args.no_oversampling and USE_OVERSAMPLING,
        use_rebalancing=not args.no_rebalancing and USE_REBALANCING,
        use_learned_dag=args.learned_dag,
        sample_size=sample_size,
        output_path=RESULTS_DIR / f"synthetic_{DATASET_NAME}_{name}.csv",
        checkpoint_path=RESULTS_DIR / f"{name}_checkpoint.pt",
    )
    synthetic, meta = run_diffusion_benchmark(cfg)

    ledger_path = RESULTS_DIR / "cdp_tabdiff_privacy_ledger.csv"
    row = pd.DataFrame([meta])
    if ledger_path.exists():
        old = pd.read_csv(ledger_path)
        old = old[old["model"] != name]
        combined = pd.concat([old, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(ledger_path, index=False)
    return synthetic


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 128
        args.embed_dim = 16
        args.n_blocks = 2
        args.timesteps = 200
        args.epsilon = 20.0
        _run_one(args, epsilon=args.epsilon, sample_size=512)
        return

    _run_one(args, epsilon=args.epsilon)
    if args.epsilon_sensitivity:
        for eps in CDP_TABDIFF_EPSILON_SENSITIVITY:
            suffix = f"_eps{int(eps)}"
            print(f"\n=== CDP-TabDiff sensitivity run (epsilon={eps}) ===")
            _run_one(args, epsilon=eps, suffix=suffix)


if __name__ == "__main__":
    main()
