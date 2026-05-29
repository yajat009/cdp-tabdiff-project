"""
CDP-TabDiff denoiser network.

A TabDDPM-style MLP backbone for tabular diffusion, with two structural
modifications:

1. **Causal masking** — every linear layer in the network is a
   :class:`CausalMaskedLinear` whose mask is derived from the
   counterfactual-fairness-aware DAG defined in :mod:`dag`. Hidden
   units are organised into per-feature blocks of width ``embed_dim``;
   block ``f`` corresponds to feature ``f``. This way the input
   projection, the residual MLP stack, and the output head all respect
   the same parent / child relations.

2. **Opacus-friendly normalisation** — only :class:`torch.nn.LayerNorm`
   is used. BatchNorm is forbidden because it couples samples and
   breaks per-sample gradients in DP-SGD.

Time conditioning follows the standard DDPM recipe: sinusoidal
embedding of the timestep ``t`` → small MLP → broadcast addition to
the hidden state at every residual block (FiLM-style additive bias).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dag import (
    FEATURE_ORDER,
    build_mask_from_assignments,
    get_causal_mask,
    make_dense_mask_like,
    make_random_mask_like,
)
from .mask import CausalMaskedLinear

MaskType = str  # one of "causal", "learned", "random", "none"
_VALID_MASK_TYPES = {"causal", "learned", "random", "none"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """How a single tabular column is represented in the network."""

    name: str
    is_categorical: bool
    cardinality: int  # 1 for continuous; >1 for categorical (one-hot width)

    @property
    def input_width(self) -> int:
        return self.cardinality if self.is_categorical else 1


def default_stroke_schema(
    df: "pd.DataFrame | None" = None,
) -> Tuple[FeatureSpec, ...]:
    """
    Return the stroke :class:`FeatureSpec` tuple inferred from data.

    When ``df`` is omitted, the local benchmark CSV is loaded and
    preprocessed so cardinalities match the real dataset rather than
    hard-coded guesses.
    """
    from .stroke_schema import default_stroke_schema as _infer_default

    return _infer_default(df)


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for the diffusion step."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Residual block with FiLM-style additive time conditioning
# ---------------------------------------------------------------------------
class CausalResidualBlock(nn.Module):
    """
    Masked MLP block with gradient highways (ReLU skip connections).

    Where input and output widths match (``hidden_dim``), the block applies
    ``out = ReLU(transform(h)) + h`` so minority-class gradients survive
    DP noise injection. LayerNorm is used throughout (BatchNorm-free for
    Opacus).
    """

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        mask: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer = CausalMaskedLinear(hidden_dim, hidden_dim, mask=mask)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = h
        transformed = self.layer(self.norm(h)) + self.time_proj(t_emb)
        return F.relu(transformed) + residual


# ---------------------------------------------------------------------------
# Main denoiser
# ---------------------------------------------------------------------------
class CDPTabDiffDenoiser(nn.Module):
    """
    MLP diffusion denoiser for tabular data with causal weight masking.

    Forward signature:

        ``x_t``: ``(B, D_in)`` noised tabular row. Column layout is the
        concatenation of each ``FeatureSpec``'s slice in the order of
        ``FEATURE_ORDER`` (one column for continuous features, a one-hot
        block of width ``cardinality`` for categoricals).

        ``t``: ``(B,)`` integer diffusion timesteps in ``[0, T)``.

    Returns the predicted noise ``eps_hat`` of the same shape as ``x_t``.
    """

    def __init__(
        self,
        schema: Sequence[FeatureSpec] | None = None,
        embed_dim: int = 32,
        n_blocks: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        mask_type: MaskType = "causal",
        random_mask_seed: int = 0,
        learned_adj: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if mask_type not in _VALID_MASK_TYPES:
            raise ValueError(
                f"mask_type must be one of {sorted(_VALID_MASK_TYPES)}; "
                f"got {mask_type!r}."
            )
        if mask_type == "learned" and learned_adj is None:
            raise ValueError(
                "mask_type='learned' requires a feature-level learned_adj tensor."
            )
        self.mask_type: MaskType = mask_type
        self.learned_adj = learned_adj
        if schema is None:
            schema = default_stroke_schema()
        self.schema: Tuple[FeatureSpec, ...] = tuple(schema)
        # Sanity check: schema must align with the DAG's feature order.
        if tuple(s.name for s in self.schema) != FEATURE_ORDER:
            raise ValueError(
                "FeatureSpec ordering must match the stroke DAG feature order; "
                f"got {tuple(s.name for s in self.schema)} vs {FEATURE_ORDER}."
            )

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.time_dim = time_dim
        self.num_features = len(self.schema)
        self.hidden_dim = self.num_features * embed_dim
        self.input_dim = sum(s.input_width for s in self.schema)

        # ------------------------------------------------------------------
        # Per-column feature-id assignments (used to build the masks).
        # ------------------------------------------------------------------
        in_feature_ids: List[int] = []
        for f_idx, spec in enumerate(self.schema):
            in_feature_ids.extend([f_idx] * spec.input_width)
        hidden_feature_ids = np.repeat(
            np.arange(self.num_features), embed_dim
        ).tolist()
        out_feature_ids = list(in_feature_ids)  # output dims mirror input layout

        # ------------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------------
        # We always build the "causal" mask first because the "random" and
        # "none" ablation variants are derived from it (same shape, same
        # per-layer density for "random"; all-ones for "none").
        mask_kw = {"learned_adj": learned_adj} if learned_adj is not None else {}
        causal_input_mask = build_mask_from_assignments(
            in_feature_ids=in_feature_ids,
            out_feature_ids=hidden_feature_ids,
            **mask_kw,
        )
        causal_hidden_mask = build_mask_from_assignments(
            in_feature_ids=hidden_feature_ids,
            out_feature_ids=hidden_feature_ids,
            **mask_kw,
        )
        causal_output_mask = build_mask_from_assignments(
            in_feature_ids=hidden_feature_ids,
            out_feature_ids=out_feature_ids,
            **mask_kw,
        )

        def _resolve(causal: torch.Tensor, layer_seed: int) -> torch.Tensor:
            if mask_type in {"causal", "learned"}:
                return causal
            if mask_type == "random":
                return make_random_mask_like(
                    causal, seed=random_mask_seed + layer_seed
                )
            if mask_type == "none":
                return make_dense_mask_like(causal)
            raise AssertionError(f"unreachable mask_type {mask_type!r}")

        input_mask = _resolve(causal_input_mask, layer_seed=0)
        output_mask = _resolve(causal_output_mask, layer_seed=1)
        # One random permutation per hidden layer (so each block gets its
        # own sparsity pattern) — keeps the comparison apples-to-apples
        # with the causal model which has a fixed structure throughout.
        hidden_masks = [
            _resolve(causal_hidden_mask, layer_seed=2 + i)
            for i in range(n_blocks)
        ]

        # ------------------------------------------------------------------
        # Time embedding stack (small MLP; no mask needed -- broadcast)
        # ------------------------------------------------------------------
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stroke_cond_embed = nn.Embedding(2, time_dim)

        # ------------------------------------------------------------------
        # Input projection (masked) + LayerNorm
        # ------------------------------------------------------------------
        self.input_proj = CausalMaskedLinear(
            self.input_dim, self.hidden_dim, mask=input_mask
        )
        # The skip connection must respect the same causal mask as input_proj;
        # a dense nn.Linear here would re-open severed pathways (e.g.
        # gender -> stroke), silently defeating the fairness masking.
        self.input_skip = CausalMaskedLinear(
            self.input_dim, self.hidden_dim, mask=input_mask
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)

        # ------------------------------------------------------------------
        # Residual stack
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(
                    hidden_dim=self.hidden_dim,
                    time_dim=time_dim,
                    mask=hidden_masks[i],
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )

        # ------------------------------------------------------------------
        # Output head (masked) -- predicts noise on the input layout
        # ------------------------------------------------------------------
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = CausalMaskedLinear(
            self.hidden_dim, self.input_dim, mask=output_mask
        )

        # Cache the column offsets so callers can slice ``x_t`` per feature.
        offsets: List[int] = [0]
        for spec in self.schema:
            offsets.append(offsets[-1] + spec.input_width)
        self.register_buffer(
            "_column_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------
    @property
    def column_offsets(self) -> List[int]:
        return self._column_offsets.tolist()

    def feature_slice(self, feature_name: str) -> slice:
        names = [s.name for s in self.schema]
        idx = names.index(feature_name)
        offsets = self.column_offsets
        return slice(offsets[idx], offsets[idx + 1])

    def per_feature_mask_density(self) -> Dict[str, float]:
        """
        Average fraction of *allowed* incoming edges for each feature's
        output units on the final projection layer.

        Values near 1.0 mean dense connectivity; near 0.0 mean heavy masking.
        """
        out = self.output_proj
        densities: Dict[str, float] = {}
        for f_idx, spec in enumerate(self.schema):
            sl = self.feature_slice(spec.name)
            block = out.mask[sl, :]
            densities[spec.name] = float(block.mean().item())
        return densities

    def log_mask_statistics(self, log_fn=print) -> Dict[str, float]:
        """Log per-feature mask density; warn when >80% of edges are masked."""
        densities = self.per_feature_mask_density()
        layers = []
        for name, mod in self.named_modules():
            if isinstance(mod, CausalMaskedLinear):
                layers.append((name, float(mod.mask.mean().item())))
        log_fn("[mask] per-layer density:")
        for name, dens in layers:
            log_fn(f"  {name}: {dens:.3f}")
        log_fn("[mask] per-feature output density (allowed-edge fraction):")
        for feat, dens in densities.items():
            masked_frac = 1.0 - dens
            flag = " OVER-MASKED" if masked_frac > 0.80 else ""
            log_fn(f"  {feat}: density={dens:.3f} masked={masked_frac:.1%}{flag}")
        return densities

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stroke_labels: Optional[torch.Tensor] = None,
        stroke_cond_drop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[1] != self.input_dim:
            raise ValueError(
                f"x_t must be (B, {self.input_dim}); got {tuple(x_t.shape)}."
            )
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"t must be shape (B,); got {tuple(t.shape)}."
            )

        if stroke_labels is None:
            stroke_labels = x_t[:, self.feature_slice("stroke")].argmax(dim=1)
        stroke_emb = self.stroke_cond_embed(stroke_labels.long())
        if stroke_cond_drop is not None:
            keep = (~stroke_cond_drop).float().unsqueeze(1)
            stroke_emb = stroke_emb * keep

        t_emb = self.time_embed(t) + stroke_emb
        h = self.input_proj(x_t)
        h = F.relu(h) + self.input_skip(x_t)
        h = self.input_norm(h)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_norm(h)
        return self.output_proj(h)
