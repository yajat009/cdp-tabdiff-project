"""Dataset loading and preprocessing for the Stroke benchmark."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from cdp_tabdiff.stroke_schema import preprocess_stroke

from .config import (
    LOCAL_CSV,
    RANDOM_STATE,
    SAMPLE_SIZE,
    TARGET_COLUMN,
)


def preprocess_stroke_df(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible alias for :func:`preprocess_stroke`."""
    return preprocess_stroke(df)


def load_stroke_data(
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load and preprocess the local Stroke CSV."""
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(
            f"Expected dataset at {LOCAL_CSV}. Please place "
            "'healthcare-dataset-stroke-data.csv' in the data/ directory."
        )

    # Treat the dataset's literal ``N/A`` (bmi only) as NaN, but keep
    # ``Unknown`` from smoking_status as a valid category.
    raw = pd.read_csv(LOCAL_CSV, na_values=["N/A"], low_memory=False)
    df = preprocess_stroke(raw)

    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    # Stratified subsample preserves the severe stroke class imbalance.
    frac = sample_size / len(df)
    parts = []
    for _, group in df.groupby(TARGET_COLUMN, group_keys=False):
        n = max(1, int(round(len(group) * frac)))
        parts.append(
            group.sample(n=min(n, len(group)), random_state=random_state)
        )
    sub = pd.concat(parts, ignore_index=True)
    return sub.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# Backwards-compatible alias for any caller still using the old name.
load_diabetes_data = load_stroke_data


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified hold-out split on the stroke target."""
    from sklearn.model_selection import train_test_split as sk_split

    train, test = sk_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
