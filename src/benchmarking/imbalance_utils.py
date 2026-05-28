"""Class-imbalance utilities for stroke synthesizer training and evaluation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    RANDOM_STATE,
    REBALANCE_RATE_MAX,
    REBALANCE_RATE_MIN,
    RESULTS_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_RATE,
    USE_REBALANCING,
)


def compute_positive_rate(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> float:
    """Return the fraction of rows where ``target_col == 1``."""
    if len(df) == 0:
        return float("nan")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    return float((y == 1).mean())


def oversample_training_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    sampling_strategy: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Oversample the minority class in *training* data only via RandomOverSampler.

    ``sampling_strategy=0.2`` sets the minority count to 20% of the majority
    count (not 20% of total rows). SMOTE is intentionally avoided for GAN
    baselines.
    """
    from imblearn.over_sampling import RandomOverSampler

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col].astype(int).values

    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = ros.fit_resample(X, y)
    out = pd.DataFrame(X_res, columns=feature_cols)
    out[target_col] = y_res.astype(int)
    return out.reset_index(drop=True)


def rebalance_synthetic(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    target_rate: float = TARGET_POSITIVE_RATE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Post-hoc rebalance synthetic data to match a target positive rate.

    Keeps as many negatives as possible, subsamples (or upsamples with
    replacement) positives so that
    ``n_positives / n_total ≈ target_rate``.
    """
    if len(df) == 0:
        return df.copy()

    pos = df[df[target_col].astype(int) == 1]
    neg = df[df[target_col].astype(int) == 0]

    neg_out = neg.copy()
    n_neg = len(neg_out)
    if n_neg == 0:
        raise ValueError("rebalance_synthetic: no negative rows to keep.")

    n_pos = max(1, int(round(n_neg * target_rate / (1.0 - target_rate))))

    if len(pos) >= n_pos:
        pos_out = pos.sample(n=n_pos, random_state=random_state)
    else:
        pos_out = pos.sample(n=n_pos, replace=True, random_state=random_state)

    out = pd.concat([neg_out, pos_out], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def assert_rebalanced_rate(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    low: float = REBALANCE_RATE_MIN,
    high: float = REBALANCE_RATE_MAX,
) -> None:
    """Raise if the rebalanced positive rate is outside the acceptable band."""
    rate = compute_positive_rate(df, target_col)
    if not (low <= rate <= high):
        raise AssertionError(
            f"Rebalanced positive rate {rate:.4f} is outside "
            f"[{low:.4f}, {high:.4f}]. Check sampling/rebalancing logic."
        )


def log_positive_rate(
    model_name: str,
    df: pd.DataFrame,
    stage: str,
    target_col: str = TARGET_COLUMN,
) -> float:
    """Log and return the positive rate at a pipeline stage."""
    rate = compute_positive_rate(df, target_col)
    print(f"[{model_name}] syn_positive_rate ({stage}): {rate:.4f} ({rate * 100:.2f}%)")
    return rate


def verify_baseline_positive_rate(
    rate: float,
    model_name: str,
    low: float = 0.02,
    high: float = 0.25,
) -> bool:
    """Return True if the synthetic positive rate is within acceptable bounds."""
    ok = low <= rate <= high
    if not ok:
        warnings.warn(
            f"[{model_name}] synthetic positive rate {rate:.4f} is outside "
            f"[{low:.2f}, {high:.2f}]. Will retry with stronger oversampling.",
            stacklevel=2,
        )
    return ok


def finalize_synthetic_output(
    synthetic: pd.DataFrame,
    model_name: str,
    *,
    use_rebalancing: bool = USE_REBALANCING,
    target_rate: float = TARGET_POSITIVE_RATE,
    random_state: int = RANDOM_STATE,
    assert_rebalanced: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cast target to int, optionally rebalance, log before/after rates, and
    persist a row to ``results/synthesis_rates.csv``.
    """
    out = synthetic.copy()
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = (
            pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    rate_before = log_positive_rate(model_name, out, "before_rebalancing")

    if use_rebalancing:
        out = rebalance_synthetic(
            out,
            target_col=TARGET_COLUMN,
            target_rate=target_rate,
            random_state=random_state,
        )
        rate_after = log_positive_rate(model_name, out, "after_rebalancing")
        if assert_rebalanced:
            assert_rebalanced_rate(out, target_col=TARGET_COLUMN)
    else:
        rate_after = rate_before

    _append_synthesis_rate_row(
        model_name=model_name,
        rate_before=rate_before,
        rate_after=rate_after,
        rebalanced=int(use_rebalancing),
    )
    return out, {"rate_before": rate_before, "rate_after": rate_after}


def _append_synthesis_rate_row(
    model_name: str,
    rate_before: float,
    rate_after: float,
    rebalanced: int,
) -> None:
    """Append one row to the shared synthesis-rate audit log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "synthesis_rates.csv"
    row = pd.DataFrame(
        [
            {
                "model": model_name,
                "syn_positive_rate_before": rate_before,
                "syn_positive_rate_after": rate_after,
                "rebalanced": rebalanced,
            }
        ]
    )
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["model"] != model_name]
        combined = pd.concat([old, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(path, index=False)


def prepare_synthesizer_training_data(
    full_df: pd.DataFrame,
    *,
    use_oversampling: bool,
    sampling_strategy: float,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the benchmark sample into train/test (test never oversampled) and
    optionally oversample the training fold only.
    """
    from .data import train_test_split

    train, test = train_test_split(full_df, random_state=random_state)
    if use_oversampling:
        train = oversample_training_data(
            train,
            target_col=TARGET_COLUMN,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)
