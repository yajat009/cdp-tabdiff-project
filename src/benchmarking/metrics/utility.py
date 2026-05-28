"""Utility metrics: Train-on-Synthetic, Test-on-Real (TSTR) for stroke."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from .encoding import build_preprocessor, encode_dataframe


def compute_tstr(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train a class-balanced RandomForest on synthetic data and evaluate on
    real held-out data. Returns downstream AUROC and macro-F1.
    """
    feature_cols = [c for c in real_train.columns if c != target_column]
    X_syn = synthetic[feature_cols]
    y_syn = synthetic[target_column].astype(int)
    X_test = real_test[feature_cols]
    y_test = real_test[target_column].astype(int)

    preprocessor, _, _ = build_preprocessor(X_syn)
    X_syn_enc = encode_dataframe(X_syn, preprocessor)
    X_test_enc = encode_dataframe(X_test, preprocessor)

    if y_syn.nunique() < 2:
        # Degenerate synthesizer: collapsed the rare positive class.
        return {"tstr_auc": float("nan"), "tstr_macro_f1": float("nan")}

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_syn_enc, y_syn)
    y_pred = clf.predict(X_test_enc)
    y_proba = clf.predict_proba(X_test_enc)[:, 1]

    auc = float(roc_auc_score(y_test, y_proba))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return {"tstr_auc": auc, "tstr_macro_f1": macro_f1}
