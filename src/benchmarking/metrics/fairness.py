"""Fairness metrics using fairlearn for the binary stroke target."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    true_positive_rate,
)
from sklearn.ensemble import RandomForestClassifier

from .encoding import build_preprocessor, encode_dataframe


def _equal_opportunity_diff(y_true, y_pred, sensitive_features) -> float:
    """TPR gap across groups (a.k.a. Equal Opportunity difference)."""
    mf = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return float(mf.difference(method="between_groups"))


def _fpr_diff(y_true, y_pred, sensitive_features) -> float:
    mf = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return float(mf.difference(method="between_groups"))


def compute_fairness(
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
    sensitive_column: str,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train on synthetic data, predict on real test data, and measure fairness
    gaps with fairlearn using ``sensitive_column`` as the protected attribute.

    Returns DP, Equal-Opportunity (TPR), FPR, and TPR gaps. The TPR and
    Equal-Opportunity gaps are identical by definition for the binary case;
    both are reported for downstream-table compatibility.
    """
    feature_cols = [
        c for c in real_test.columns if c not in {target_column, sensitive_column}
    ]

    X_syn = synthetic[feature_cols]
    X_test = real_test[feature_cols]

    y_syn = synthetic[target_column].astype(int)
    y_test = real_test[target_column].astype(int)
    sensitive = real_test[sensitive_column].astype(str)

    preprocessor, _, _ = build_preprocessor(X_syn)
    X_syn_enc = encode_dataframe(X_syn, preprocessor)
    X_test_enc = encode_dataframe(X_test, preprocessor)

    # Handle the degenerate case where a synthesizer collapsed the rare
    # positive class entirely (severe imbalance failure mode).
    syn_pos_rate = float(y_syn.mean())

    if y_syn.nunique() < 2:
        return {
            "demographic_parity_diff": float("nan"),
            "equal_opportunity_diff": float("nan"),
            "fpr_diff": float("nan"),
            "tpr_diff": float("nan"),
            "syn_positive_rate": syn_pos_rate,
            "pred_positive_rate": float("nan"),
            "mode_collapse_flag": 1.0,
        }

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_syn_enc, y_syn)
    y_pred = clf.predict(X_test_enc)
    pred_pos_rate = float(y_pred.mean())
    # Mode-collapse heuristic: the downstream classifier predicts the same
    # class for >=99% of the held-out test rows, which makes parity-style
    # gaps trivially small and unsafe to report at face value.
    collapsed = 1.0 if (pred_pos_rate <= 0.01 or pred_pos_rate >= 0.99) else 0.0

    dpd = float(
        demographic_parity_difference(
            y_test, y_pred, sensitive_features=sensitive
        )
    )
    tpr_gap = _equal_opportunity_diff(y_test, y_pred, sensitive)
    fpr_gap = _fpr_diff(y_test, y_pred, sensitive)
    # Sanity reference: equalized_odds_difference = max(TPR gap, FPR gap).
    _ = float(
        equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)
    )

    return {
        "demographic_parity_diff": dpd,
        "equal_opportunity_diff": tpr_gap,
        "fpr_diff": fpr_gap,
        "tpr_diff": tpr_gap,
        "syn_positive_rate": syn_pos_rate,
        "pred_positive_rate": pred_pos_rate,
        "mode_collapse_flag": collapsed,
    }
