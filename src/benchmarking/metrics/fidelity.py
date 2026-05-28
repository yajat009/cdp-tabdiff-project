"""Fidelity metrics: correlation error and random 3-way marginal MAE."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .encoding import label_encode_dataframe


def compute_pairwise_correlation_error(
    real_df: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    """Mean absolute error between pairwise correlation matrices."""
    feature_cols = [c for c in real_df.columns if c in synthetic.columns]
    real_enc = label_encode_dataframe(real_df[feature_cols])
    syn_enc = label_encode_dataframe(synthetic[feature_cols])

    corr_real = real_enc.corr()
    corr_syn = syn_enc.corr()

    mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
    diff = np.abs(corr_real.values[mask] - corr_syn.values[mask])
    return float(np.nanmean(diff))


def _joint_distribution(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalized joint frequency table keyed by ``columns``."""
    counts = (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="prob")
    )
    counts["prob"] = counts["prob"] / counts["prob"].sum()
    return counts


def compute_random_3way_marginal_mae(
    real_df: pd.DataFrame,
    synthetic: pd.DataFrame,
    n_triplets: int = 50,
    random_state: int = 42,
) -> float:
    """
    Average MAE between real and synthetic joint distributions over random
    3-column marginals.
    """
    rng = np.random.default_rng(random_state)
    columns = [c for c in real_df.columns if c in synthetic.columns]
    if len(columns) < 3:
        return float("nan")

    errors = []
    for _ in range(n_triplets):
        triplet = list(rng.choice(columns, size=3, replace=False))
        real_dist = _joint_distribution(real_df, triplet)
        syn_dist = _joint_distribution(synthetic, triplet)
        aligned = real_dist.merge(
            syn_dist,
            on=triplet,
            how="outer",
            suffixes=("_real", "_syn"),
        ).fillna(0.0)
        errors.append(
            float(np.mean(np.abs(aligned["prob_real"] - aligned["prob_syn"])))
        )

    return float(np.mean(errors))


def compute_fidelity(
    real_df: pd.DataFrame,
    synthetic: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute all fidelity metrics."""
    return {
        "pairwise_correlation_error": compute_pairwise_correlation_error(
            real_df, synthetic
        ),
        "random_3way_marginal_mae": compute_random_3way_marginal_mae(
            real_df, synthetic, random_state=random_state
        ),
    }
