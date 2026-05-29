"""
Round-trip encoder between the Stroke dataframe and the CDP-TabDiff tensor
layout.

Continuous columns (``age``, ``avg_glucose_level``, ``bmi``) are
standard-scaled to zero-mean / unit-variance. Categorical columns are
one-hot encoded to a width equal to their ``FeatureSpec.cardinality``.
The encoder fixes the category ordering at ``fit`` time so the inverse
transform is deterministic.

The resulting tensor has columns in the exact order of
:data:`EXPECTED_FEATURE_COLUMNS` and slot widths matching
:meth:`CDPTabDiffDenoiser.column_offsets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .stroke_schema import (
    EXPECTED_FEATURE_COLUMNS as FEATURE_ORDER,
    infer_stroke_schema,
)
from .model import FeatureSpec, default_stroke_schema


@dataclass
class _ColumnState:
    spec: FeatureSpec
    # For continuous: mean / std and clipping bounds from training data.
    mean: float = 0.0
    std: float = 1.0
    z_min: float = -4.0
    z_max: float = 4.0
    raw_min: float = 0.0
    raw_max: float = 1.0
    categories: List[str] = field(default_factory=list)


class StrokeEncoder:
    """Fits on a Stroke dataframe, returns a ``(N, D)`` float tensor."""

    def __init__(
        self,
        schema: Sequence[FeatureSpec] | None = None,
        clip_zscore: float = 4.0,
        reference_df: pd.DataFrame | None = None,
    ) -> None:
        if schema is not None:
            self.schema: Tuple[FeatureSpec, ...] = tuple(schema)
        elif reference_df is not None:
            self.schema = infer_stroke_schema(reference_df)
        else:
            self.schema = default_stroke_schema()
        self.clip_zscore = clip_zscore
        if tuple(s.name for s in self.schema) != FEATURE_ORDER:
            raise ValueError(
                "Encoder schema order must match the stroke feature order."
            )
        self.states: Dict[str, _ColumnState] = {}
        self._input_dim: int = sum(s.input_width for s in self.schema)
        self._fitted: bool = False

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def column_offsets(self) -> List[int]:
        offsets = [0]
        for s in self.schema:
            offsets.append(offsets[-1] + s.input_width)
        return offsets

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "StrokeEncoder":
        missing = [s.name for s in self.schema if s.name not in df.columns]
        if missing:
            raise KeyError(f"missing columns in dataframe: {missing}")

        for spec in self.schema:
            col = df[spec.name]
            state = _ColumnState(spec=spec)
            if spec.is_categorical:
                # Sorted level ordering so the one-hot index layout is
                # deterministic across runs AND identical to the schema
                # inference in ``infer_stroke_schema`` (which also sorts).
                # First-seen ordering depended on post-shuffle row order and
                # could silently flip the stroke one-hot block, inverting the
                # class label seen by the model. See encoding/schema parity.
                seen: List[str] = sorted(col.astype(str).unique().tolist())
                if len(seen) > spec.cardinality:
                    raise ValueError(
                        f"column '{spec.name}' has {len(seen)} distinct "
                        f"values; schema cardinality is {spec.cardinality}. "
                        "Either widen the schema or pre-filter rare values."
                    )
                # Pad to cardinality with synthetic placeholders so the
                # one-hot width is always exactly ``cardinality``.
                while len(seen) < spec.cardinality:
                    seen.append(f"__pad_{len(seen)}__")
                state.categories = seen
            else:
                values = pd.to_numeric(col, errors="coerce").to_numpy(
                    dtype=np.float64
                )
                state.mean = float(np.nanmean(values))
                state.std = float(np.nanstd(values) + 1e-8)
                z = (values - state.mean) / state.std
                z = z[~np.isnan(z)]
                if len(z):
                    # Clip generated z-scores to the training support (with
                    # margin) but never exceed a global TabDDPM-style bound.
                    state.z_min = float(max(np.min(z) - 0.5, -self.clip_zscore))
                    state.z_max = float(min(np.max(z) + 0.5, self.clip_zscore))
                state.raw_min = float(np.nanmin(values))
                state.raw_max = float(np.nanmax(values))
            self.states[spec.name] = state

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("StrokeEncoder.fit must be called first.")
        n = len(df)
        out = np.zeros((n, self._input_dim), dtype=np.float32)
        offsets = self.column_offsets
        for f_idx, spec in enumerate(self.schema):
            state = self.states[spec.name]
            slot = slice(offsets[f_idx], offsets[f_idx + 1])
            col = df[spec.name]
            if spec.is_categorical:
                idx_map = {c: i for i, c in enumerate(state.categories)}
                vals = col.astype(str).to_numpy()
                idx = np.array(
                    [idx_map.get(v, 0) for v in vals], dtype=np.int64
                )
                one_hot = np.zeros((n, spec.cardinality), dtype=np.float32)
                one_hot[np.arange(n), idx] = 1.0
                out[:, slot] = one_hot
            else:
                values = pd.to_numeric(col, errors="coerce").to_numpy(
                    dtype=np.float32
                )
                values = np.where(np.isnan(values), state.mean, values)
                out[:, slot] = ((values - state.mean) / state.std).reshape(
                    -1, 1
                )
        return torch.from_numpy(out)

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------
    def inverse_transform(self, arr: torch.Tensor | np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("StrokeEncoder.fit must be called first.")
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if arr.ndim != 2 or arr.shape[1] != self._input_dim:
            raise ValueError(
                f"inverse_transform expects (N, {self._input_dim}); "
                f"got {arr.shape}."
            )

        offsets = self.column_offsets
        out: Dict[str, np.ndarray] = {}
        for f_idx, spec in enumerate(self.schema):
            state = self.states[spec.name]
            slot = slice(offsets[f_idx], offsets[f_idx + 1])
            block = arr[:, slot]
            if spec.is_categorical:
                # Argmax of the (noised) one-hot block; ignore __pad__ buckets.
                logits = np.clip(block, -20.0, 20.0)
                for j, cat in enumerate(state.categories):
                    if cat.startswith("__pad_"):
                        logits[:, j] = -np.inf
                idx = np.argmax(logits, axis=1)
                values = np.array(
                    [state.categories[i] for i in idx], dtype=object
                )
                out[spec.name] = values
            else:
                z = np.clip(block.reshape(-1), state.z_min, state.z_max)
                values = z * state.std + state.mean
                values = np.clip(values, state.raw_min, state.raw_max)
                out[spec.name] = values
        df = pd.DataFrame(out, columns=[s.name for s in self.schema])
        # Post-cast integer-valued categoricals back to int when natural.
        for spec in self.schema:
            if spec.is_categorical:
                vals = df[spec.name]
                if all(self._is_intlike(v) for v in vals.unique()):
                    df[spec.name] = vals.astype(int)
        return df

    @staticmethod
    def _is_intlike(v: object) -> bool:
        try:
            f = float(v)
            return float(int(f)) == f
        except (TypeError, ValueError):
            return False
