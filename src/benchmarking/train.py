"""Universal training wrapper covering SynthCity and SDV synthesizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from .config import (
    DATASET_NAME,
    OVERSAMPLE_STRATEGY,
    PLUGIN_KWARGS,
    RANDOM_STATE,
    RESULTS_DIR,
    SDV_MODELS,
    SENSITIVE_COLUMN,
    SYNTHCITY_MODELS,
    TARGET_COLUMN,
    USE_OVERSAMPLING,
)
from .data import load_stroke_data
from .imbalance_utils import (
    compute_positive_rate,
    log_positive_rate,
    prepare_synthesizer_training_data,
    verify_baseline_positive_rate,
)


def configure_gpu() -> str:
    """Select compute device and log GPU availability."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] Using {name} ({torch.cuda.device_count()} device(s))")
    else:
        print("[GPU] CUDA not available — training on CPU")
    return device


# ---------------------------------------------------------------------------
# SynthCity branch
# ---------------------------------------------------------------------------
def _train_synthcity(
    model_name: str,
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    n_samples: int,
) -> pd.DataFrame:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    # Conditional generation on ``stroke`` via target_column metadata.
    loader = GenericDataLoader(
        df,
        target_column=TARGET_COLUMN,
        sensitive_features=[SENSITIVE_COLUMN],
    )
    plugin = Plugins().get(model_name, **kwargs)
    print(f"Training synthcity:{model_name} (conditional on {TARGET_COLUMN})...")
    plugin.fit(loader)
    print(f"Generating {n_samples:,} synthetic rows...")
    return plugin.generate(count=n_samples).dataframe()


# ---------------------------------------------------------------------------
# SDV branch
# ---------------------------------------------------------------------------
_SDV_SYNTHESIZERS = {
    "sdv_ctgan": "CTGANSynthesizer",
    "sdv_tvae": "TVAESynthesizer",
    "sdv_gaussian_copula": "GaussianCopulaSynthesizer",
    "sdv_copula_gan": "CopulaGANSynthesizer",
}


def _train_sdv(
    model_name: str,
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    n_samples: int,
) -> pd.DataFrame:
    from sdv.metadata import SingleTableMetadata
    from sdv import single_table as sdv_single

    cls_name = _SDV_SYNTHESIZERS[model_name]
    SynthCls = getattr(sdv_single, cls_name)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    if TARGET_COLUMN in df.columns:
        metadata.update_column(column_name=TARGET_COLUMN, sdtype="categorical")

    synth_kwargs = dict(kwargs)
    enforce_min_max = synth_kwargs.pop("enforce_min_max_values", True)
    synthesizer = SynthCls(
        metadata,
        enforce_min_max_values=enforce_min_max,
        **synth_kwargs,
    )
    print(f"Training sdv:{cls_name} (enforce_min_max_values={enforce_min_max})...")
    synthesizer.fit(df)
    print(f"Sampling {n_samples:,} synthetic rows...")
    return synthesizer.sample(num_rows=n_samples)


def _cast_target_int(synthetic: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in synthetic.columns:
        synthetic = synthetic.copy()
        synthetic[TARGET_COLUMN] = (
            pd.to_numeric(synthetic[TARGET_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    return synthetic


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def train_and_generate(
    model_name: str,
    plugin_kwargs: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
    output_path: Optional[Path] = None,
    use_oversampling: bool = USE_OVERSAMPLING,
    oversample_strategy: float = OVERSAMPLE_STRATEGY,
) -> pd.DataFrame:
    """
    Train a synthesizer (SynthCity or SDV) and save the synthetic CSV to
    ``results/synthetic_stroke_{model_name}.csv``. Returns the dataframe.

    Training uses the *train* fold only (test fold is never touched).
    When ``use_oversampling`` is True, the minority class in the train
    fold is oversampled via RandomOverSampler before fitting.
    """
    configure_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(PLUGIN_KWARGS.get(model_name, {}))
    if plugin_kwargs:
        kwargs.update(plugin_kwargs)
    if model_name in SYNTHCITY_MODELS:
        kwargs.setdefault("device", device)

    print(f"\n=== Benchmark: {model_name.upper()} ===")
    print(f"Plugin kwargs: {kwargs}")
    print(f"use_oversampling={use_oversampling}, strategy={oversample_strategy}")

    full_df = (
        load_stroke_data(sample_size=sample_size)
        if sample_size
        else load_stroke_data()
    )
    print(f"Benchmark sample shape: {full_df.shape}")
    print(
        f"Real positive rate (full sample): "
        f"{compute_positive_rate(full_df, TARGET_COLUMN):.4f}"
    )

    n_samples = len(full_df)
    synthetic: pd.DataFrame | None = None
    strategy = oversample_strategy

    for attempt in range(2):
        train_df, _ = prepare_synthesizer_training_data(
            full_df,
            use_oversampling=use_oversampling,
            sampling_strategy=strategy,
            random_state=RANDOM_STATE,
        )
        print(
            f"Training fold shape: {train_df.shape} | "
            f"positive rate: {compute_positive_rate(train_df, TARGET_COLUMN):.4f}"
        )

        if model_name in SYNTHCITY_MODELS:
            synthetic = _train_synthcity(model_name, train_df, kwargs, n_samples)
        elif model_name in SDV_MODELS:
            synthetic = _train_sdv(model_name, train_df, kwargs, n_samples)
        else:
            raise ValueError(
                f"Unknown model '{model_name}'. Expected one of "
                f"{sorted(SYNTHCITY_MODELS + SDV_MODELS)}."
            )

        synthetic = _cast_target_int(synthetic)
        rate = log_positive_rate(model_name, synthetic, "raw")
        if verify_baseline_positive_rate(rate, model_name):
            break
        if attempt == 0 and use_oversampling:
            strategy = min(oversample_strategy * 2.0, 0.5)
            print(
                f"Retrying {model_name} with doubled oversampling "
                f"strategy={strategy:.2f}..."
            )
        else:
            print(
                f"WARNING: {model_name} still outside target positive-rate band "
                f"after retry; saving best-effort output."
            )
            break

    assert synthetic is not None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = RESULTS_DIR / f"synthetic_{DATASET_NAME}_{model_name}.csv"
    synthetic.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")

    train_cache = RESULTS_DIR / "real_train_reference.csv"
    if not train_cache.exists():
        full_df.to_csv(train_cache, index=False)
        print(f"Cached real training reference to {train_cache}")

    return synthetic
