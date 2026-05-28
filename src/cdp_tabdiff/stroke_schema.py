"""
Stroke dataset schema and causal-structure helpers.

Column names, cardinalities, and preprocessing expectations are derived from
the benchmark's preprocessed Kaggle Stroke CSV rather than duplicated as
magic numbers in :mod:`dag` or :mod:`model`.

The causal edge template encodes domain knowledge about stroke risk factors.
At training time we validate that every edge references real columns and log
empirical association scores from the training fold so unsupported edges are
visible in the logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Columns dropped during benchmark preprocessing (see ``benchmarking.data``).
DROP_COLUMNS: Tuple[str, ...] = ("id",)
VALID_GENDER_VALUES: Tuple[str, ...] = ("Male", "Female")

PROTECTED_ATTRIBUTE: str = "gender"
TARGET: str = "stroke"

# Canonical column order after preprocessing (matches ``load_stroke_data``).
EXPECTED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
)

# Treat these numeric columns as categorical binary targets/indicators.
CATEGORICAL_BY_NAME: Tuple[str, ...] = (
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
    "stroke",
)

CONTINUOUS_BY_NAME: Tuple[str, ...] = ("age", "avg_glucose_level", "bmi")

# Domain causal template for the stroke SCM (parent -> child edges).
# Values must reference ``EXPECTED_FEATURE_COLUMNS``.
STROKE_DAG_TEMPLATE: Dict[str, List[str]] = {
    "gender": [],
    "age": [],
    "Residence_type": [],
    "ever_married": ["age"],
    "work_type": ["age", "ever_married"],
    "smoking_status": ["age", "work_type"],
    "bmi": ["age", "gender"],
    "avg_glucose_level": ["age", "bmi", "gender"],
    "hypertension": [
        "age",
        "gender",
        "bmi",
        "avg_glucose_level",
        "smoking_status",
    ],
    "heart_disease": [
        "age",
        "gender",
        "bmi",
        "avg_glucose_level",
        "smoking_status",
        "hypertension",
    ],
    "stroke": [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "avg_glucose_level",
        "smoking_status",
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def preprocess_stroke(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the benchmark preprocessing pipeline so schema inference and the
    DAG always see the same column set as ``benchmarking.data.load_stroke_data``.
    """
    out = df.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()

    if "bmi" in out.columns:
        out["bmi"] = pd.to_numeric(out["bmi"], errors="coerce")
        out["bmi"] = out["bmi"].fillna(out["bmi"].median())

    if "gender" in out.columns:
        out = out[out["gender"].isin(VALID_GENDER_VALUES)].copy()

    out = out.dropna().reset_index(drop=True)
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype(int)
    return out


def load_reference_stroke_dataframe(
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and preprocess the local Stroke CSV used by the benchmark."""
    if csv_path is None:
        csv_path = _project_root() / "data" / "healthcare-dataset-stroke-data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. Place "
            "'healthcare-dataset-stroke-data.csv' in the data/ directory."
        )
    raw = pd.read_csv(csv_path, na_values=["N/A"], low_memory=False)
    return preprocess_stroke(raw)


def infer_feature_order(df: pd.DataFrame) -> Tuple[str, ...]:
    """Return benchmark column order, validating the preprocessed dataframe."""
    missing = [c for c in EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe is missing expected stroke columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    extra = [c for c in df.columns if c not in EXPECTED_FEATURE_COLUMNS]
    if extra:
        raise KeyError(
            f"Unexpected extra columns {extra}; expected exactly "
            f"{list(EXPECTED_FEATURE_COLUMNS)}."
        )
    return EXPECTED_FEATURE_COLUMNS


def is_categorical_column(name: str, series: pd.Series) -> bool:
    """Decide representation type using column name and observed values."""
    if name in CONTINUOUS_BY_NAME:
        return False
    if name in CATEGORICAL_BY_NAME:
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return int(series.nunique(dropna=True)) <= 10


def infer_stroke_schema(df: pd.DataFrame) -> Tuple["FeatureSpec", ...]:
    """
    Build :class:`FeatureSpec` entries from an observed preprocessed dataframe.

    Categorical cardinalities come from the unique values present in ``df``
    (after preprocessing), not from hard-coded guesses.
    """
    from .model import FeatureSpec

    order = infer_feature_order(df)
    specs: List[FeatureSpec] = []
    for name in order:
        col = df[name]
        if is_categorical_column(name, col):
            categories = sorted(col.astype(str).unique().tolist())
            cardinality = len(categories)
            if cardinality < 2:
                raise ValueError(
                    f"Column '{name}' has cardinality {cardinality}; "
                    "expected at least 2 levels."
                )
            specs.append(FeatureSpec(name, True, cardinality))
        else:
            specs.append(FeatureSpec(name, False, 1))
    return tuple(specs)


def default_stroke_schema(
    df: Optional[pd.DataFrame] = None,
) -> Tuple["FeatureSpec", ...]:
    """
    Return the stroke schema inferred from ``df`` or the local reference CSV.
    """
    if df is None:
        df = load_reference_stroke_dataframe()
    return infer_stroke_schema(df)


def build_stroke_dag(
    feature_names: Sequence[str] = EXPECTED_FEATURE_COLUMNS,
) -> Dict[str, List[str]]:
    """Instantiate the causal template for the given feature set."""
    names = tuple(feature_names)
    if set(names) != set(STROKE_DAG_TEMPLATE):
        raise ValueError(
            "feature_names must match the stroke DAG template keys; "
            f"got {set(names)} vs {set(STROKE_DAG_TEMPLATE)}"
        )
    order = {n: i for i, n in enumerate(names)}
    out: Dict[str, List[str]] = {}
    for child, parents in STROKE_DAG_TEMPLATE.items():
        if child not in order:
            raise KeyError(f"DAG child '{child}' not in feature_names.")
        out[child] = [p for p in parents if p in order]
    return out


def validate_dataframe_for_stroke(df: pd.DataFrame) -> None:
    """Raise if ``df`` is not a valid preprocessed stroke table."""
    infer_feature_order(df)
    if not set(df[PROTECTED_ATTRIBUTE].astype(str)).issubset(set(VALID_GENDER_VALUES)):
        bad = set(df[PROTECTED_ATTRIBUTE].astype(str)) - set(VALID_GENDER_VALUES)
        raise ValueError(f"Invalid gender levels after preprocessing: {bad}")
    if TARGET in df.columns and df[TARGET].nunique() < 2:
        raise ValueError(
            f"Target '{TARGET}' must contain both classes for training; "
            f"got counts {df[TARGET].value_counts().to_dict()}."
        )


def _edge_association(parent: pd.Series, child: pd.Series) -> float:
    """Simple association score in [0, 1] for logging DAG edge support."""
    p = parent.astype(str)
    c = child.astype(str)
    if p.nunique() <= 1 or c.nunique() <= 1:
        return 0.0

    if pd.api.types.is_numeric_dtype(parent) and parent.nunique() > 10:
        # Continuous parent: correlation magnitude (monotone proxy).
        pn = pd.to_numeric(parent, errors="coerce")
        if pd.api.types.is_numeric_dtype(child) and child.nunique() > 10:
            cn = pd.to_numeric(child, errors="coerce")
            corr = pn.corr(cn)
            return float(abs(corr)) if corr == corr else 0.0
        # ANOVA-style eta: between-group variance / total variance.
        groups = [pn[c == level].dropna() for level in c.unique()]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return 0.0
        overall = pn.dropna()
        if len(overall) == 0:
            return 0.0
        grand = overall.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((overall - grand) ** 2).sum()
        return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0

    # Categorical parent: Cramér's V.
    table = pd.crosstab(p, c)
    if table.size == 0:
        return 0.0
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0:
        return 0.0
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - expected) ** 2 / np.where(expected > 0, expected, np.nan))
    r, k = obs.shape
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def summarize_dag_data_support(
    df: pd.DataFrame,
    dag: Dict[str, List[str]],
    *,
    weak_threshold: float = 0.02,
    log_fn=print,
) -> Dict[Tuple[str, str], float]:
    """
    Score each template edge against ``df`` and log weakly supported edges.

    Returns a mapping ``(parent, child) -> association score``.
    """
    validate_dataframe_for_stroke(df)
    scores: Dict[Tuple[str, str], float] = {}
    log_fn("[dag] empirical edge support on training data:")
    for child, parents in dag.items():
        for parent in parents:
            score = _edge_association(df[parent], df[child])
            scores[(parent, child)] = score
            flag = " WEAK" if score < weak_threshold else ""
            log_fn(f"  {parent} -> {child}: {score:.3f}{flag}")
    return scores


# Import-time validation of the template against expected columns.
assert set(STROKE_DAG_TEMPLATE) == set(EXPECTED_FEATURE_COLUMNS), (
    "STROKE_DAG_TEMPLATE keys must match EXPECTED_FEATURE_COLUMNS."
)
for _child, _parents in STROKE_DAG_TEMPLATE.items():
    for _p in _parents:
        if _p not in STROKE_DAG_TEMPLATE:
            raise ValueError(f"Unknown parent '{_p}' for child '{_child}'.")
