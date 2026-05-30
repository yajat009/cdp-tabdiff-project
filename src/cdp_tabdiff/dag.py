"""
Causal prior for the Kaggle Stroke dataset — precision-pruned masks.

Structural metadata lives in :mod:`stroke_schema`. Mask construction uses
*precision pruning*: keep all clinical predictors wired directly to
``stroke``, and sever only explicit unfair pathways from the protected
attribute ``gender`` (directly to ``stroke`` and to known proxies such
as ``work_type``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from .stroke_schema import (
    EXPECTED_FEATURE_COLUMNS,
    PROTECTED_ATTRIBUTE,
    TARGET,
    build_stroke_dag,
    summarize_dag_data_support,
    validate_dataframe_for_stroke,
)

# Re-export canonical names used throughout the package.
FEATURE_ORDER: Tuple[str, ...] = EXPECTED_FEATURE_COLUMNS
STROKE_DAG: Dict[str, List[str]] = build_stroke_dag(FEATURE_ORDER)

# Clinical / lifestyle variables that may influence stroke directly.
STROKE_CLINICAL_PARENTS: Tuple[str, ...] = (
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
)

# Sever gender -> child edges for the outcome and gender-linked proxies only.
GENDER_SEVERED_CHILDREN: Set[str] = {TARGET, "work_type"}

# Temporal / structural roles used when orienting a learned CPDAG.
EXOGENOUS_NODES: Set[str] = {PROTECTED_ATTRIBUTE, "age", "Residence_type"}
SINK_NODES: Set[str] = {TARGET}


# ---------------------------------------------------------------------------
# Causal discovery (PC algorithm + fairness scalpel)
# ---------------------------------------------------------------------------
def encode_dataframe_for_pc(
    dataframe: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Encode the Stroke table for conditional-independence testing.

    * Categorical columns → ordinal codes (stable sorted levels).
    * Continuous columns → z-scored numerics.

    Returns ``(n_samples, n_features)`` float64 matrix in ``feature_names`` order.
    """
    from .stroke_schema import (
        CONTINUOUS_BY_NAME,
        infer_feature_order,
        is_categorical_column,
        validate_dataframe_for_stroke,
    )

    validate_dataframe_for_stroke(dataframe)
    order = infer_feature_order(dataframe)
    if feature_names is not None and tuple(order) != tuple(feature_names):
        raise ValueError(
            "feature_names must match the preprocessed stroke column order."
        )

    block = dataframe[list(order)].copy()
    out_cols: List[np.ndarray] = []
    for name in order:
        col = block[name]
        if is_categorical_column(name, col):
            levels = sorted(col.astype(str).unique().tolist())
            idx_map = {level: i for i, level in enumerate(levels)}
            encoded = col.astype(str).map(idx_map).astype(np.float64).to_numpy()
            out_cols.append(encoded)
        elif name in CONTINUOUS_BY_NAME:
            values = pd.to_numeric(col, errors="coerce").astype(np.float64).to_numpy()
            mu = np.nanmean(values)
            sigma = np.nanstd(values) + 1e-8
            values = np.where(np.isnan(values), mu, values)
            out_cols.append((values - mu) / sigma)
        else:
            values = pd.to_numeric(col, errors="coerce").astype(np.float64).to_numpy()
            out_cols.append(values)
    matrix = np.column_stack(out_cols).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("PC encoding produced non-finite values; check preprocessing.")
    return matrix, tuple(order)


def _discretize_for_chisq(
    matrix: np.ndarray,
    n_bins: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Bin continuous columns so the chi-square CI test can be applied."""
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
        subsample=int(min(10_000, matrix.shape[0])),
        random_state=random_state,
    )
    return discretizer.fit_transform(matrix).astype(np.int64)


def _run_pc(
    matrix: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "auto",
    random_state: int = 42,
) -> np.ndarray:
    """
    Run the PC algorithm and return the causal-learn adjacency matrix ``G``.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "causal-learn is required for discover_learned_dag. "
            "Install with `pip install causal-learn>=0.1.4.7`."
        ) from exc

    test = indep_test.lower()
    if test == "auto":
        test = "chisq_binned" if matrix.shape[0] < 500 else "fisherz"
    if test == "fisherz":
        cg = pc(matrix, alpha=alpha, indep_test="fisherz", show_progress=False)
    elif test in {"chisq", "chisq_binned"}:
        discrete = _discretize_for_chisq(matrix, random_state=random_state)
        cg = pc(discrete, alpha=alpha, indep_test="chisq", show_progress=False)
    else:
        raise ValueError(
            f"Unknown indep_test '{indep_test}'. Use auto, fisherz, or chisq_binned."
        )
    return np.asarray(cg.G.graph)


def _preferred_parent_child(
    i: int,
    j: int,
    feature_names: Sequence[str],
    expert_dag: Dict[str, List[str]],
) -> Tuple[int, int]:
    """Return ``(parent_idx, child_idx)`` for an undirected CPDAG edge."""
    ni, nj = feature_names[i], feature_names[j]

    if nj in EXOGENOUS_NODES and ni not in EXOGENOUS_NODES:
        return j, i
    if ni in EXOGENOUS_NODES and nj not in EXOGENOUS_NODES:
        return i, j
    if nj == TARGET and ni != TARGET:
        return i, j
    if ni == TARGET and nj != TARGET:
        return j, i
    if ni in expert_dag.get(nj, []):
        return i, j
    if nj in expert_dag.get(ni, []):
        return j, i
    if FEATURE_ORDER.index(ni) < FEATURE_ORDER.index(nj):
        return i, j
    return j, i


def cpdag_to_adjacency(
    cpdag: np.ndarray,
    feature_names: Sequence[str] = FEATURE_ORDER,
    expert_dag: Dict[str, List[str]] | None = None,
) -> np.ndarray:
    """
    Convert a causal-learn CPDAG matrix into a directed adjacency.

    Convention: ``adj[child, parent] = 1`` iff ``parent -> child``.
    """
    expert_dag = expert_dag or STROKE_DAG
    p = len(feature_names)
    adj = np.zeros((p, p), dtype=np.float32)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cpdag[i, j] == -1 and cpdag[j, i] == 1:
                # i --> j: parent i, child j
                adj[j, i] = 1.0
            elif cpdag[i, j] == 1 and cpdag[j, i] == -1:
                # j --> i: parent j, child i
                adj[i, j] = 1.0

    for i in range(p):
        for j in range(i + 1, p):
            if cpdag[i, j] == -1 and cpdag[j, i] == -1:
                parent, child = _preferred_parent_child(
                    i, j, feature_names, expert_dag
                )
                adj[child, parent] = 1.0

    return adj


def _enforce_temporal_constraints(
    adj: np.ndarray,
    feature_names: Sequence[str] = FEATURE_ORDER,
) -> np.ndarray:
    """Hard orientations: exogenous nodes have no parents; stroke has no children."""
    out = adj.copy()
    for idx, name in enumerate(feature_names):
        if name in EXOGENOUS_NODES:
            out[idx, :] = 0.0
        if name in SINK_NODES:
            out[:, idx] = 0.0
    return out


def apply_fairness_scalpel(
    adj: np.ndarray,
    feature_names: Sequence[str] = FEATURE_ORDER,
    protected_attribute: str = PROTECTED_ATTRIBUTE,
    target: str = TARGET,
) -> np.ndarray:
    """
    Remove unfair directed edges from ``gender`` to ``stroke`` and known proxies.
    """
    if protected_attribute not in feature_names or target not in feature_names:
        raise KeyError("protected_attribute and target must be in feature_names.")

    out = adj.copy()
    prot_idx = feature_names.index(protected_attribute)
    target_idx = feature_names.index(target)

    out[target_idx, prot_idx] = 0.0
    for child_name in GENDER_SEVERED_CHILDREN:
        if child_name in feature_names:
            out[feature_names.index(child_name), prot_idx] = 0.0

    proxy_nodes = descendants_of(protected_attribute) & set(feature_names)
    proxy_nodes.discard(target)
    for child_name in proxy_nodes:
        out[feature_names.index(child_name), prot_idx] = 0.0

    return out


def build_expert_adjacency(
    feature_names: Sequence[str] = FEATURE_ORDER,
) -> np.ndarray:
    """Binary adjacency from the hand-crafted precision-pruned expert graph."""
    p = len(feature_names)
    adj = np.zeros((p, p), dtype=np.float32)
    for child_idx, child_name in enumerate(feature_names):
        for parent_idx, parent_name in enumerate(feature_names):
            if is_allowed_edge(parent_name, child_name):
                adj[child_idx, parent_idx] = 1.0
    return adj


def merge_adjacency(
    expert_adj: np.ndarray,
    learned_adj: np.ndarray,
    mode: str = "union",
) -> np.ndarray:
    """Combine expert and learned adjacency matrices."""
    if expert_adj.shape != learned_adj.shape:
        raise ValueError("expert_adj and learned_adj must share the same shape.")
    if mode == "union":
        return np.clip(expert_adj + learned_adj, 0.0, 1.0).astype(np.float32)
    if mode == "intersect":
        return (expert_adj * learned_adj).astype(np.float32)
    if mode == "learned":
        return learned_adj.astype(np.float32)
    raise ValueError(f"Unknown merge mode '{mode}'.")


def dag_dict_from_adjacency(
    adj: np.ndarray,
    feature_names: Sequence[str] = FEATURE_ORDER,
) -> Dict[str, List[str]]:
    """Convert ``adj[child, parent]=1`` into a parents-of dictionary."""
    parents: Dict[str, List[str]] = {name: [] for name in feature_names}
    for child_idx, child_name in enumerate(feature_names):
        for parent_idx, parent_name in enumerate(feature_names):
            if child_idx == parent_idx:
                continue
            if adj[child_idx, parent_idx] >= 0.5:
                parents[child_name].append(parent_name)
    return parents


def discover_learned_dag(
    dataframe: pd.DataFrame,
    protected_attribute: str = PROTECTED_ATTRIBUTE,
    target: str = TARGET,
    *,
    alpha: float = 0.05,
    indep_test: str = "auto",
    random_state: int = 42,
    apply_fairness: bool = True,
    merge_with_expert: bool = False,
    merge_mode: str = "union",
    log_fn=print,
) -> torch.Tensor:
    """
    Learn a directed causal adjacency matrix with the PC algorithm.

    Pipeline
    --------
    1. Encode mixed Stroke columns for CI testing.
    2. Run PC (``causallearn.search.ConstraintBased.PC``).
    3. Orient CPDAG edges with temporal rules (age/gender exogenous, stroke sink).
    4. Apply the fairness scalpel (remove ``gender -> stroke`` and proxy paths).
    5. Optionally union with the expert adjacency.

    Returns
    -------
    torch.Tensor
        Feature-level mask of shape ``(n_features, n_features)`` where
        ``mask[child, parent] = 1`` iff ``parent -> child``. Plug into
        :func:`build_mask_from_learned_adjacency` / :class:`CausalMaskedLinear`.
    """
    matrix, feature_names = encode_dataframe_for_pc(dataframe)
    cpdag = _run_pc(
        matrix,
        alpha=alpha,
        indep_test=indep_test,
        random_state=random_state,
    )
    adj = cpdag_to_adjacency(cpdag, feature_names=feature_names)
    adj = _enforce_temporal_constraints(adj, feature_names=feature_names)
    if apply_fairness:
        adj = apply_fairness_scalpel(
            adj,
            feature_names=feature_names,
            protected_attribute=protected_attribute,
            target=target,
        )
    if merge_with_expert:
        expert_adj = build_expert_adjacency(feature_names)
        adj = merge_adjacency(expert_adj, adj, mode=merge_mode)
        adj = apply_fairness_scalpel(
            adj,
            feature_names=feature_names,
            protected_attribute=protected_attribute,
            target=target,
        )

    if log_fn:
        n_edges = int(adj.sum())
        prot_idx = feature_names.index(protected_attribute)
        tgt_idx = feature_names.index(target)
        log_fn(
            f"[PC] learned DAG: {n_edges} directed edges among "
            f"{len(feature_names)} features | "
            f"gender->stroke={adj[tgt_idx, prot_idx]:.0f} "
            f"density={adj.mean():.1%}"
        )
    return torch.from_numpy(adj.astype(np.float32))


def build_mask_from_learned_adjacency(
    learned_adj: torch.Tensor,
    in_feature_ids: Sequence[int],
    out_feature_ids: Sequence[int],
    allow_self_loop: bool = True,
) -> torch.Tensor:
    """
    Expand a feature-level learned adjacency into a unit-level CausalMaskedLinear mask.
    """
    in_ids = np.asarray(in_feature_ids, dtype=np.int64)
    out_ids = np.asarray(out_feature_ids, dtype=np.int64)
    adj = learned_adj.detach().cpu().numpy()
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("learned_adj must be a square (F, F) matrix.")

    mask = np.zeros((len(out_ids), len(in_ids)), dtype=np.float32)
    for o, fo in enumerate(out_ids):
        for i, fi in enumerate(in_ids):
            if allow_self_loop and fo == fi:
                mask[o, i] = 1.0
            elif adj[int(fo), int(fi)] >= 0.5:
                mask[o, i] = 1.0
    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------
def _validate_dag(dag: Dict[str, List[str]]) -> None:
    """Sanity-check the SCM (keys, unknown parents, acyclicity)."""
    assert set(dag) == set(FEATURE_ORDER), (
        "DAG keys must match FEATURE_ORDER exactly."
    )
    parents = {k: list(v) for k, v in dag.items()}
    in_degree = {k: len(v) for k, v in parents.items()}
    children: Dict[str, List[str]] = {k: [] for k in parents}
    for node, ps in parents.items():
        for p in ps:
            if p not in parents:
                raise ValueError(f"Unknown parent '{p}' for node '{node}'.")
            children[p].append(node)
    queue = [n for n, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if visited != len(parents):
        raise ValueError("STROKE_DAG contains a cycle.")


_validate_dag(STROKE_DAG)


def log_dag_data_support(df, *, log_fn=print) -> Dict[Tuple[str, str], float]:
    """Log empirical association scores for each DAG edge on ``df``."""
    validate_dataframe_for_stroke(df)
    return summarize_dag_data_support(df, STROKE_DAG, log_fn=log_fn)


def ancestors_of(node: str, dag: Dict[str, List[str]] | None = None) -> Set[str]:
    """Return the set of strict ancestors of ``node`` (excluding itself)."""
    dag = dag or STROKE_DAG
    out: Set[str] = set()
    stack = list(dag[node])
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(dag[p])
    return out


def descendants_of(node: str, dag: Dict[str, List[str]] | None = None) -> Set[str]:
    """Return the set of strict descendants of ``node`` (excluding itself)."""
    dag = dag or STROKE_DAG
    children: Dict[str, List[str]] = {k: [] for k in dag}
    for n, ps in dag.items():
        for p in ps:
            children[p].append(n)
    out: Set[str] = set()
    stack = list(children[node])
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(children[c])
    return out


def is_allowed_edge(in_feature: str, out_feature: str) -> bool:
    """
    Edge-level precision pruning for counterfactual fairness.

    Default: allow the connection (dense mask, ~80%+ alive globally).
    Forbidden explicitly:
      * gender -> stroke
      * gender -> work_type (proxy path)
    For ``stroke`` outputs, only clinical/lifestyle parents (+ self) remain.
    """
    if out_feature == TARGET:
        if in_feature == PROTECTED_ATTRIBUTE:
            return False
        return in_feature in STROKE_CLINICAL_PARENTS or in_feature == TARGET
    if out_feature == "work_type" and in_feature == PROTECTED_ATTRIBUTE:
        return False
    return True


def allowed_parents(node: str, dag: Dict[str, List[str]] | None = None) -> Set[str]:
    """
    Parent *set* view of :func:`is_allowed_edge` (used by diagnostics/tests).

    For ``stroke``, returns clinical parents only. For other nodes, returns
    every feature that is allowed to connect into ``node``.
    """
    _ = dag  # template retained for data-support logging elsewhere
    if node == TARGET:
        return set(STROKE_CLINICAL_PARENTS)
    allowed = {node}
    for candidate in FEATURE_ORDER:
        if is_allowed_edge(candidate, node):
            allowed.add(candidate)
    return allowed


def mask_density(mask: torch.Tensor) -> float:
    """Fraction of allowed (non-zero) connections in a binary mask."""
    if mask.numel() == 0:
        return float("nan")
    return float(mask.mean().item())


# ---------------------------------------------------------------------------
# Mask construction (MADE-style)
# ---------------------------------------------------------------------------
def default_feature_assignment(width: int, num_features: int) -> np.ndarray:
    """Round-robin assignment of units to feature ids (length == width)."""
    if width <= 0:
        raise ValueError("width must be positive.")
    return np.arange(width) % num_features


def build_mask_from_assignments(
    in_feature_ids: Sequence[int],
    out_feature_ids: Sequence[int],
    feature_names: Sequence[str] = FEATURE_ORDER,
    allow_self_loop: bool = True,
    dag: Dict[str, List[str]] | None = None,
    learned_adj: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build a binary ``(out_dim, in_dim)`` mask from feature assignments.

    When ``learned_adj`` is provided, edges come from the learned feature-level
    adjacency (``learned_adj[child, parent] = 1``). Otherwise the expert
    precision-pruned DAG via :func:`is_allowed_edge` is used.
    """
    in_ids = np.asarray(in_feature_ids, dtype=np.int64)
    out_ids = np.asarray(out_feature_ids, dtype=np.int64)
    F = len(feature_names)
    if in_ids.max(initial=-1) >= F or out_ids.max(initial=-1) >= F:
        raise ValueError("feature id out of range for given feature_names.")

    if learned_adj is not None:
        return build_mask_from_learned_adjacency(
            learned_adj,
            in_ids,
            out_ids,
            allow_self_loop=allow_self_loop,
        )

    dag = dag or STROKE_DAG
    in_ids = np.asarray(in_feature_ids, dtype=np.int64)
    out_ids = np.asarray(out_feature_ids, dtype=np.int64)
    F = len(feature_names)
    if in_ids.max(initial=-1) >= F or out_ids.max(initial=-1) >= F:
        raise ValueError("feature id out of range for given feature_names.")

    allowed_ids: List[Set[int]] = []
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for j, out_name in enumerate(feature_names):
        ids = {
            name_to_idx[in_name]
            for in_name in feature_names
            if is_allowed_edge(in_name, out_name)
        }
        if allow_self_loop:
            ids.add(j)
        allowed_ids.append(ids)

    mask = np.zeros((len(out_ids), len(in_ids)), dtype=np.float32)
    for o, fo in enumerate(out_ids):
        legal = allowed_ids[int(fo)]
        row_mask = np.isin(in_ids, list(legal))
        mask[o] = row_mask.astype(np.float32)
    return torch.from_numpy(mask)


def make_random_mask_like(
    causal_mask: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    """Permute the positions of ones while preserving mask density."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    flat = causal_mask.reshape(-1)
    n_ones = int(flat.sum().item())
    perm = torch.randperm(flat.numel(), generator=g)
    new = torch.zeros_like(flat)
    new[perm[:n_ones]] = 1.0
    return new.reshape_as(causal_mask).to(causal_mask.dtype)


def make_dense_mask_like(causal_mask: torch.Tensor) -> torch.Tensor:
    """All-ones mask matching ``causal_mask``'s shape (no structure)."""
    return torch.ones_like(causal_mask)


def get_causal_mask(
    input_dim: int,
    output_dim: int,
    feature_indices: Optional[Iterable[int]] = None,
    in_feature_ids: Optional[Iterable[int]] = None,
    out_feature_ids: Optional[Iterable[int]] = None,
    feature_names: Sequence[str] = FEATURE_ORDER,
    allow_self_loop: bool = True,
    dag: Dict[str, List[str]] | None = None,
    learned_adj: torch.Tensor | None = None,
    log_density: bool = False,
    log_fn=print,
) -> torch.Tensor:
    """
    Build a precision-pruned ``(output_dim, input_dim)`` binary mask.

    ``feature_indices`` is a convenience alias that sets both ``in_feature_ids``
    and ``out_feature_ids`` to the same per-unit assignment (typical for
    square feature-to-feature layers).

    When ``log_density=True``, logs the fraction of retained connections.
    Precision pruning targets ~80% density on square feature masks.
    """
    F = len(feature_names)

    if feature_indices is not None:
        in_feature_ids = feature_indices
        out_feature_ids = feature_indices

    if in_feature_ids is None:
        in_ids = default_feature_assignment(input_dim, F)
    else:
        in_ids = np.asarray(list(in_feature_ids), dtype=np.int64)
        if len(in_ids) != input_dim:
            raise ValueError("len(in_feature_ids) must equal input_dim.")

    if out_feature_ids is None:
        out_ids = default_feature_assignment(output_dim, F)
    else:
        out_ids = np.asarray(list(out_feature_ids), dtype=np.int64)
        if len(out_ids) != output_dim:
            raise ValueError("len(out_feature_ids) must equal output_dim.")

    mask = build_mask_from_assignments(
        in_ids,
        out_ids,
        feature_names=feature_names,
        allow_self_loop=allow_self_loop,
        dag=dag,
        learned_adj=learned_adj,
    )

    if log_density:
        dens = mask_density(mask)
        label = "learned" if learned_adj is not None else "precision-pruned"
        log_fn(
            f"[dag] {label} mask density={dens:.1%} "
            f"shape=({output_dim}, {input_dim})"
        )
    return mask
