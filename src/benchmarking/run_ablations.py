"""
Ablation runner for CDP-TabDiff.

Trains and samples five variants of the model so we can isolate the
contribution of (a) the causal structural prior, (b) the DP-SGD
privacy budget, and (c) generic mask sparsity:

  +-------------------+-----------+-----------+-----------------------+
  | Variant           | mask_type | dp_enabled| Tests                 |
  +===================+===========+===========+=======================+
  | base_tabddpm      | none      | False     | Utility ceiling       |
  | fair_tabdiff      | causal    | False     | Cost of fairness only |
  | dp_tabddpm_dense  | none      | True      | DP failure mode       |
  | dp_tabddpm_random | random    | True      | Sparsity != structure |
  | cdp_tabdiff       | causal    | True      | Our hero model        |
  +-------------------+-----------+-----------+-----------------------+

Each run writes:
  * ``results/synthetic_stroke_<variant>.csv``
  * ``results/<variant>_checkpoint.pt``
  * one row in ``results/ablation_privacy_ledger.csv`` (with epsilon=inf
    for non-DP variants).

The CSV filename pattern matches the regex used by
``evaluate_all.discover_synthetic_files`` so the downstream metric
suite picks the five files up automatically alongside the existing
baselines, provided the variant names are listed in
``benchmarking.config.MODELS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence

import _bootstrap  # noqa: F401

import pandas as pd

from benchmarking.config import (
    DATASET_NAME,
    RANDOM_STATE,
    RESULTS_DIR,
    STROKE_DIM_LOSS_WEIGHT,
    STROKE_SAMPLE_LOSS_WEIGHT,
    USE_OVERSAMPLING,
    USE_REBALANCING,
)
from benchmarking.diffusion_runner import DiffusionRunConfig, run_diffusion_benchmark


# ---------------------------------------------------------------------------
# Variant spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AblationSpec:
    name: str          # also the suffix on synthetic_stroke_<name>.csv
    mask_type: str     # "causal" | "random" | "none"
    dp_enabled: bool


ABLATIONS: Sequence[AblationSpec] = (
    AblationSpec("base_tabddpm",      "none",   False),
    AblationSpec("fair_tabdiff",      "causal", False),
    AblationSpec("dp_tabddpm_dense",  "none",   True),
    AblationSpec("dp_tabddpm_random", "random", True),
    AblationSpec("cdp_tabdiff",       "causal", True),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stroke-dim-weight", type=float, default=STROKE_DIM_LOSS_WEIGHT)
    p.add_argument("--stroke-sample-weight", type=float, default=STROKE_SAMPLE_LOSS_WEIGHT)
    p.add_argument(
        "--variant",
        choices=[s.name for s in ABLATIONS] + ["all"],
        default="all",
        help="Run a single variant, or all five.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (subsample, 2 epochs, lax DP budget) for sanity checks.",
    )
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
    p.add_argument("--random-mask-seed", type=int, default=RANDOM_STATE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        epochs = 2
        batch_size = 128
        embed_dim = 16
        n_blocks = 2
        timesteps = 100
        target_epsilon = 20.0
        sample_size: Optional[int] = 512
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        embed_dim = args.embed_dim
        n_blocks = args.n_blocks
        timesteps = args.timesteps
        target_epsilon = args.epsilon
        sample_size = None

    chosen: List[AblationSpec] = (
        list(ABLATIONS)
        if args.variant == "all"
        else [s for s in ABLATIONS if s.name == args.variant]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger_rows: List[dict] = []

    for spec in chosen:
        cfg = DiffusionRunConfig(
            model_name=spec.name,
            mask_type=spec.mask_type,
            dp_enabled=spec.dp_enabled,
            epochs=epochs,
            batch_size=batch_size,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            timesteps=timesteps,
            schedule_kind=args.schedule,
            target_epsilon=target_epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            lr=args.lr,
            stroke_dim_loss_weight=args.stroke_dim_weight,
            stroke_sample_loss_weight=args.stroke_sample_weight,
            use_oversampling=not args.no_oversampling and USE_OVERSAMPLING,
            use_rebalancing=not args.no_rebalancing and USE_REBALANCING,
            random_mask_seed=args.random_mask_seed,
            sample_size=sample_size,
            output_path=RESULTS_DIR / f"synthetic_{DATASET_NAME}_{spec.name}.csv",
            checkpoint_path=RESULTS_DIR / f"{spec.name}_checkpoint.pt",
        )
        try:
            _, meta = run_diffusion_benchmark(cfg)
            ledger_rows.append(
                {
                    "variant": spec.name,
                    "mask_type": spec.mask_type,
                    "dp_enabled": int(spec.dp_enabled),
                    "epsilon": meta["epsilon"],
                    "delta": meta["delta"],
                    "max_grad_norm": meta["max_grad_norm"],
                    "epochs": meta["epochs"],
                    "timesteps": meta["timesteps"],
                    "syn_positive_rate": meta["syn_positive_rate_after"],
                    "syn_positive_rate_before": meta["syn_positive_rate_before"],
                    "synthetic_path": meta["synthetic_path"],
                    "checkpoint_path": meta["checkpoint_path"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] variant '{spec.name}' failed: {exc}")
            import traceback as _tb

            _tb.print_exc()

    if ledger_rows:
        ledger_path = RESULTS_DIR / "ablation_privacy_ledger.csv"
        if ledger_path.exists():
            old = pd.read_csv(ledger_path)
            new = pd.DataFrame(ledger_rows)
            combined = pd.concat([old, new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["variant"], keep="last")
        else:
            combined = pd.DataFrame(ledger_rows)
        combined.to_csv(ledger_path, index=False)
        print(f"\nLedger written to {ledger_path}")
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
