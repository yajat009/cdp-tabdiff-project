"""Evaluate all synthetic outputs and produce summary tables (Stroke)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

import _bootstrap  # noqa: F401

from benchmarking.config import (
    DATASET_NAME,
    MODELS,
    RANDOM_STATE,
    RESULTS_DIR,
    SENSITIVE_COLUMN,
    TARGET_COLUMN,
)
from benchmarking.data import load_stroke_data, train_test_split
from benchmarking.metrics import (
    compute_dcr,
    compute_fairness,
    compute_fidelity,
    compute_mia,
    compute_tstr,
)

METRIC_COLUMNS = [
    "model",
    # Utility
    "tstr_auc",
    "tstr_macro_f1",
    # Fairness
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "fpr_diff",
    "tpr_diff",
    # Privacy
    "dcr",
    "mia_auc",
    "mia_success_rate",
    # Fidelity
    "random_3way_marginal_mae",
    "pairwise_correlation_error",
    # Diagnostics for interpreting fairness/utility under class imbalance
    "syn_positive_rate_before",
    "syn_positive_rate",
    "pred_positive_rate",
    "mode_collapse_flag",
]


def discover_synthetic_files(
    results_dir: Path, models: tuple[str, ...] | None = None
) -> Dict[str, Path]:
    """Map model name -> CSV path for generated synthetic datasets."""
    pattern = re.compile(
        rf"^synthetic_{DATASET_NAME}_(?P<model>[a-z0-9_]+)\.csv$"
    )
    found: Dict[str, Path] = {}
    for path in sorted(results_dir.glob(f"synthetic_{DATASET_NAME}_*.csv")):
        match = pattern.match(path.name)
        if match:
            model = match.group("model")
            if models is None or model in models:
                found[model] = path
    return found


def evaluate_model(
    model_name: str,
    synthetic_path: Path,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
) -> Dict[str, float]:
    """Compute all benchmark metrics for one synthetic dataset."""
    synthetic = pd.read_csv(synthetic_path, low_memory=False)

    metrics: Dict[str, float] = {"model": model_name}
    metrics.update(
        compute_tstr(real_train, real_test, synthetic, TARGET_COLUMN, RANDOM_STATE)
    )
    metrics.update(
        compute_fairness(
            real_test,
            synthetic,
            TARGET_COLUMN,
            SENSITIVE_COLUMN,
            RANDOM_STATE,
        )
    )
    metrics["dcr"] = compute_dcr(real_train, synthetic)
    metrics.update(
        compute_mia(real_train, real_test, synthetic, RANDOM_STATE)
    )
    metrics.update(
        compute_fidelity(real_train, synthetic, random_state=RANDOM_STATE)
    )
    return metrics


def format_latex_table(summary: pd.DataFrame) -> str:
    """Render a booktabs-style LaTeX table for the paper."""
    display_cols = [c for c in METRIC_COLUMNS if c in summary.columns]
    df = summary[display_cols].copy()

    rename = {
        "model": "Model",
        "tstr_auc": "AUROC",
        "tstr_macro_f1": "Macro-F1",
        "demographic_parity_diff": "DP gap",
        "equal_opportunity_diff": "EO gap",
        "fpr_diff": "FPR gap",
        "tpr_diff": "TPR gap",
        "dcr": "DCR",
        "mia_auc": "MIA AUC",
        "mia_success_rate": "MIA Acc",
        "random_3way_marginal_mae": "3-way MAE",
        "pairwise_correlation_error": "Corr. Err.",
        "syn_positive_rate_before": "Syn +rate (raw)",
        "syn_positive_rate": "Syn +rate",
        "pred_positive_rate": "Pred +rate",
        "mode_collapse_flag": "Collapse",
    }
    df = df.rename(columns=rename)

    for col in df.columns:
        if col == "Model":
            continue
        df[col] = df[col].map(lambda x: "--" if pd.isna(x) else f"{x:.4f}")

    col_spec = "l" + "r" * (len(df.columns) - 1)
    header = " & ".join(df.columns) + r" \\"
    rows = [" & ".join(row.astype(str)) + r" \\ " for _, row in df.iterrows()]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic data benchmark on the Healthcare Stroke dataset "
        r"(SynthCity vs. SDV).}",
        r"\label{tab:synth_benchmark_stroke}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    real = load_stroke_data()
    real_train, real_test = train_test_split(real, random_state=RANDOM_STATE)
    print(
        f"Real train/test: {real_train.shape} / {real_test.shape} | "
        f"positive rate train={real_train[TARGET_COLUMN].mean():.4f} "
        f"test={real_test[TARGET_COLUMN].mean():.4f}"
    )

    synthetic_files = discover_synthetic_files(RESULTS_DIR, models=MODELS)
    if not synthetic_files:
        raise FileNotFoundError(
            f"No synthetic CSVs matching 'synthetic_{DATASET_NAME}_*.csv' "
            f"found in {RESULTS_DIR}. Run the benchmark_*.py scripts first."
        )

    rows: List[Dict[str, float]] = []
    for model_name, path in synthetic_files.items():
        print(f"Evaluating {model_name} ({path.name})...")
        rows.append(evaluate_model(model_name, path, real_train, real_test))

    summary = pd.DataFrame(rows)
    rates_path = RESULTS_DIR / "synthesis_rates.csv"
    if rates_path.exists():
        rates = pd.read_csv(rates_path).set_index("model")
        summary["syn_positive_rate_before"] = summary["model"].map(
            rates["syn_positive_rate_before"]
        )
    # Keep metric column order stable; tolerate any missing metrics.
    ordered = [c for c in METRIC_COLUMNS if c in summary.columns]
    summary = summary[ordered]

    summary_path = RESULTS_DIR / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    latex = format_latex_table(summary)
    latex_path = RESULTS_DIR / "results_summary.tex"
    latex_path.write_text(latex)
    print(f"Saved LaTeX table to {latex_path}")
    print("\n--- LaTeX preview ---")
    print(latex)


if __name__ == "__main__":
    main()
