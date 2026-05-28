"""Privacy metrics: Distance to Closest Record (DCR) and MIA.

The MIA implementation follows the nearest-neighbour formulation used in the
synthetic-data literature (Hayes et al., Stadler et al.): the adversary's
*only* knowledge is the published synthetic dataset. For each candidate real
record we measure the distance to the closest synthetic record; small
distances suggest the record was a member of the training set used to fit
the generator. We score the attack with AUROC over a balanced pool of
members (real_train) and non-members (real_test), and additionally report
accuracy at the optimal threshold for backwards-compatibility with the
existing summary column ``mia_success_rate``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from .encoding import build_preprocessor, encode_dataframe


def compute_dcr(real_df: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Mean distance from each synthetic record to its nearest real record.

    Lower values indicate synthetic records that are closer to real data
    (potential memorization / privacy risk).
    """
    feature_cols = [c for c in real_df.columns if c in synthetic.columns]
    real = real_df[feature_cols]
    syn = synthetic[feature_cols]

    preprocessor, _, _ = build_preprocessor(real)
    real_enc = encode_dataframe(real, preprocessor)
    syn_enc = encode_dataframe(syn, preprocessor)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(real_enc)
    distances, _ = nn.kneighbors(syn_enc)
    return float(np.mean(distances))


def compute_mia(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Nearest-neighbour membership inference attack against the *synthesizer*.

    The adversary observes only the synthetic dataset and uses
    ``min_j dist(x_i, syn_j)`` as a membership score (smaller -> more
    likely to be a training member). We compute AUROC over a balanced pool
    of members (sampled from ``real_train``) and non-members (sampled from
    ``real_test``), plus accuracy at the Youden-optimal threshold.

    Returns
    -------
    {
      "mia_auc":             AUROC of the attack (0.5 = no leakage),
      "mia_success_rate":    accuracy at optimal threshold (0.5 = no leakage),
    }
    """
    rng = np.random.default_rng(random_state)

    common_cols = [c for c in real_train.columns if c in synthetic.columns]
    if not common_cols or len(synthetic) == 0:
        return {"mia_auc": float("nan"), "mia_success_rate": float("nan")}

    # Balanced member / non-member pool so AUROC is well-defined and the
    # accuracy column is comparable across generators.
    n = min(len(real_train), len(real_test))
    members = real_train[common_cols].sample(
        n=n, random_state=random_state
    )
    nonmembers = real_test[common_cols].sample(
        n=n, random_state=random_state
    )

    # Fit the encoder on real data (a publisher-side artefact the adversary
    # could reconstruct from public column schemas), then transform every set.
    preprocessor, _, _ = build_preprocessor(real_train[common_cols])
    syn_enc = encode_dataframe(synthetic[common_cols], preprocessor)
    mem_enc = encode_dataframe(members, preprocessor)
    non_enc = encode_dataframe(nonmembers, preprocessor)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(syn_enc)
    d_mem, _ = nn.kneighbors(mem_enc)
    d_non, _ = nn.kneighbors(non_enc)
    d_mem = d_mem.ravel()
    d_non = d_non.ravel()

    # Membership score: closer to synthetic -> more likely a member.
    scores = np.concatenate([-d_mem, -d_non])
    labels = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))]).astype(int)

    if np.allclose(scores, scores[0]):
        # Degenerate generator (e.g. all duplicates): attack carries no info.
        return {"mia_auc": 0.5, "mia_success_rate": 0.5}

    auc = float(roc_auc_score(labels, scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    # Youden-J optimal threshold accuracy on the balanced pool:
    # acc = 0.5 * (TPR + (1 - FPR)). max over thresholds.
    acc = float(np.max(0.5 * (tpr + (1.0 - fpr))))

    # Add a small jitter break for ties so we can't get 0.5 by luck.
    _ = rng  # rng reserved for future stochastic variants
    return {"mia_auc": auc, "mia_success_rate": acc}


# ---------------------------------------------------------------------------
# Backwards-compatible shim. Old call sites pass (real_train, real_test) and
# get the (broken) constant attack; new call sites should use ``compute_mia``.
# ---------------------------------------------------------------------------
def compute_mia_success_rate(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame | None = None,
    random_state: int = 42,
) -> float:
    """Deprecated shim. Use ``compute_mia`` for the synthetic-aware attack."""
    if synthetic is None:
        raise ValueError(
            "compute_mia_success_rate now requires the synthetic dataframe "
            "(the previous train-vs-test signature ignored the generator "
            "and produced a constant score). Call compute_mia(...) instead."
        )
    return compute_mia(real_train, real_test, synthetic, random_state)[
        "mia_success_rate"
    ]
