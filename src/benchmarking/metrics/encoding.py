"""Feature encoding utilities shared across metric modules."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical feature lists."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    df: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a sklearn preprocessor (one-hot) for tree-model metrics."""
    numeric, categorical = infer_column_types(df)
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.fit(df)
    return preprocessor, numeric, categorical


def encode_dataframe(
    df: pd.DataFrame, preprocessor: ColumnTransformer
) -> np.ndarray:
    """Transform a dataframe into a numeric matrix."""
    return preprocessor.transform(df)


def label_encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode categoricals for fast correlation / fidelity metrics."""
    out = df.copy()
    numeric, categorical = infer_column_types(out)
    if numeric:
        out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        out[categorical] = encoder.fit_transform(out[categorical].astype(str))
    return out
