"""
Smoke tests for CDP-TabDiff.

Run as a script:

    python -m cdp_tabdiff.smoke_test
    python src/cdp_tabdiff/smoke_test.py     # via _bootstrap-style sys.path

The tests are intentionally lightweight (synthetic data, 2 epochs, 50
timesteps) and verify:

1. DAG / mask construction is consistent and the protected pathway
   ``gender -> stroke`` (direct or proxy) is severed.
2. The masked Linear layer zeroes forbidden weight entries and keeps
   them zero after backprop.
3. The denoiser produces gradients with the expected sparsity pattern.
4. The Opacus-wrapped trainer can run an end-to-end DP step + sampling
   pass without errors.
5. The encoder is a true round-trip on synthetic data.

A non-zero exit code indicates failure.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch

# Allow `python src/cdp_tabdiff/smoke_test.py` from the repo root.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from cdp_tabdiff import (  # noqa: E402
    CDPTabDiffDenoiser,
    CDPTabDiffTrainer,
    DiffusionSchedule,
    FEATURE_ORDER,
    PROTECTED_ATTRIBUTE,
    TARGET,
    StrokeEncoder,
    infer_stroke_schema,
    allowed_parents,
    descendants_of,
    get_causal_mask,
)
from cdp_tabdiff.mask import CausalMaskedLinear  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic mini dataset (matches the schema; does not need real data)
# ---------------------------------------------------------------------------
def _make_fake_dataframe(n: int = 256, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=n),
            "age": rng.uniform(20, 85, size=n),
            "hypertension": rng.integers(0, 2, size=n),
            "heart_disease": rng.integers(0, 2, size=n),
            "ever_married": rng.choice(["Yes", "No"], size=n),
            "work_type": rng.choice(
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                size=n,
            ),
            "Residence_type": rng.choice(["Urban", "Rural"], size=n),
            "avg_glucose_level": rng.uniform(55, 270, size=n),
            "bmi": rng.uniform(15, 50, size=n),
            "smoking_status": rng.choice(
                ["formerly smoked", "never smoked", "smokes", "Unknown"], size=n
            ),
            "stroke": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_pc_causal_discovery() -> None:
    """PC discovery runs on synthetic data and respects fairness constraints."""
    from cdp_tabdiff.dag import discover_learned_dag

    df = _make_fake_dataframe(n=200, seed=7)
    try:
        adj = discover_learned_dag(
            df,
            alpha=0.05,
            indep_test="fisherz",
            merge_with_expert=True,
            log_fn=None,
        )
    except ImportError:
        print("[SKIP] pc_causal_discovery (causal-learn not installed)")
        return
    assert adj.shape == (len(FEATURE_ORDER), len(FEATURE_ORDER))
    stroke_idx = FEATURE_ORDER.index(TARGET)
    gender_idx = FEATURE_ORDER.index(PROTECTED_ATTRIBUTE)
    assert adj[stroke_idx, gender_idx].item() == 0.0
    assert adj[stroke_idx, :].sum().item() > 0.0, "stroke should retain parents"


def test_learned_denoiser_forward() -> None:
    """Denoiser accepts a PC-learned adjacency without shape errors."""
    from cdp_tabdiff.dag import discover_learned_dag

    df = _make_fake_dataframe(n=256, seed=8)
    try:
        learned_adj = discover_learned_dag(df, log_fn=None, merge_with_expert=True)
    except ImportError:
        print("[SKIP] learned_denoiser_forward (causal-learn not installed)")
        return
    schema = infer_stroke_schema(df)
    model = CDPTabDiffDenoiser(
        schema=schema,
        embed_dim=8,
        n_blocks=1,
        time_dim=16,
        mask_type="learned",
        learned_adj=learned_adj,
    )
    x = torch.randn(4, model.input_dim)
    t = torch.randint(0, 50, size=(4,))
    out = model(x, t)
    assert out.shape == x.shape


def test_stroke_conditioning_separates_features() -> None:
    """After brief training, stroke=1 rows should differ from stroke=0 on age."""
    from torch.utils.data import DataLoader, TensorDataset

    df = _make_fake_dataframe(n=512, seed=9)
    schema = infer_stroke_schema(df)
    enc = StrokeEncoder(reference_df=df).fit(df)
    x = enc.transform(df)
    model = CDPTabDiffDenoiser(
        schema=schema, embed_dim=8, n_blocks=2, time_dim=32
    )
    sched = DiffusionSchedule.make(num_timesteps=20, schedule="cosine", device="cpu")
    trainer = CDPTabDiffTrainer(
        model=model,
        schedule=sched,
        device="cpu",
        p_uncond=0.0,
        use_stroke_conditioning=True,
        use_stroke_train_inpainting=True,
        cfg_guidance_scale=2.0,
        use_stroke_loss_reweighting=True,
    )
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    trainer.model.train()
    optim = torch.optim.AdamW(trainer.model.parameters(), lr=5e-3)
    for _ in range(15):
        for (batch,) in loader:
            labels = trainer._stroke_labels_from_x0(batch)
            optim.zero_grad(set_to_none=True)
            loss = trainer._diffusion_loss(batch, stroke_labels=labels)
            loss.backward()
            optim.step()

    samples = trainer.generate_samples(
        num_samples=128,
        batch_size=64,
        target_positive_rate=0.25,
    )
    syn = enc.inverse_transform(samples)
    pos = syn[syn.stroke == 1]
    neg = syn[syn.stroke == 0]
    assert len(pos) > 0 and len(neg) > 0
    age_gap = float(pos["age"].mean() - neg["age"].mean())
    assert age_gap > 0.5, f"stroke=1 not older after conditioning: gap={age_gap:.2f}"


def test_dag_severance() -> None:
    """Stroke must not depend on gender; clinical parents must remain."""
    from cdp_tabdiff.dag import STROKE_CLINICAL_PARENTS

    forbidden = {PROTECTED_ATTRIBUTE}
    parents = allowed_parents(TARGET)
    overlap = parents & forbidden
    assert not overlap, (
        f"stroke depends on forbidden ancestors via gender's closure: {overlap}"
    )
    assert set(STROKE_CLINICAL_PARENTS).issubset(parents), (
        f"missing clinical parents for stroke: "
        f"{set(STROKE_CLINICAL_PARENTS) - parents}"
    )


def test_feature_mask_shape_and_zeros() -> None:
    """Identity-assignment mask of size FxF respects the DAG row for stroke."""
    F = len(FEATURE_ORDER)
    mask = get_causal_mask(
        F, F,
        in_feature_ids=range(F),
        out_feature_ids=range(F),
    )
    assert mask.shape == (F, F)
    stroke_idx = FEATURE_ORDER.index(TARGET)
    gender_idx = FEATURE_ORDER.index(PROTECTED_ATTRIBUTE)
    assert mask[stroke_idx, gender_idx].item() == 0.0, (
        "gender -> stroke edge present despite fairness constraint."
    )
    from cdp_tabdiff.dag import STROKE_CLINICAL_PARENTS

    for clinical in STROKE_CLINICAL_PARENTS:
        c_idx = FEATURE_ORDER.index(clinical)
        assert mask[stroke_idx, c_idx].item() == 1.0, (
            f"clinical parent '{clinical}' -> stroke severed by over-pruning."
        )
    for desc in descendants_of(PROTECTED_ATTRIBUTE):
        if desc in STROKE_CLINICAL_PARENTS or desc == TARGET:
            continue
        d_idx = FEATURE_ORDER.index(desc)
        assert mask[stroke_idx, d_idx].item() == 0.0, (
            f"proxy edge {desc} -> stroke present despite fairness constraint."
        )


def test_masked_linear_zeros_forbidden_weights() -> None:
    """Forbidden weight entries are at 0 every time the layer is used."""
    torch.manual_seed(0)
    F_ = len(FEATURE_ORDER)
    mask = get_causal_mask(
        F_, F_,
        in_feature_ids=range(F_),
        out_feature_ids=range(F_),
    )
    layer = CausalMaskedLinear(F_, F_, mask=mask)
    assert torch.all((layer.weight * (1 - layer.mask)).abs() < 1e-12)

    optim = torch.optim.SGD(layer.parameters(), lr=1e-1)
    x = torch.randn(8, F_)
    y = layer(x).sum()
    y.backward()
    optim.step()
    # After the step, the *underlying* weight may have moved on forbidden
    # entries (vanilla autograd produced non-zero grads). The forward
    # pre-hook must snap them back to zero on the next call:
    _ = layer(x)
    forbidden_now = (layer.weight * (1 - layer.mask)).abs().max().item()
    assert forbidden_now == 0.0, (
        "Forbidden weight nonzero after pre-hook fired."
    )


def test_denoiser_forward_runs() -> None:
    """End-to-end forward through the masked denoiser produces valid output."""
    torch.manual_seed(0)
    df = _make_fake_dataframe(n=32, seed=4)
    schema = infer_stroke_schema(df)
    model = CDPTabDiffDenoiser(
        schema=schema, embed_dim=8, n_blocks=2, time_dim=32
    )
    B = 4
    x = torch.randn(B, model.input_dim)
    t = torch.randint(0, 100, size=(B,))
    out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    # Every masked layer must have its forbidden entries at 0 after the
    # pre-hook runs (verified by triggering one more forward call).
    _ = model(x, t)
    for name, mod in model.named_modules():
        if isinstance(mod, CausalMaskedLinear):
            forbidden_w = (mod.weight * (1 - mod.mask)).abs().max().item()
            assert forbidden_w == 0.0, f"forbidden weight nonzero in '{name}'."


def test_encoder_roundtrip() -> None:
    df = _make_fake_dataframe(n=64, seed=1)
    enc = StrokeEncoder(reference_df=df).fit(df)
    x = enc.transform(df)
    df2 = enc.inverse_transform(x)
    # Categorical columns must round-trip exactly (one-hot is invertible).
    for col in (
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
        "stroke",
    ):
        same = (df[col].astype(str).values == df2[col].astype(str).values).mean()
        assert same > 0.99, f"categorical round-trip failed for '{col}': {same:.3f}"
    # Continuous columns: standard-scale inverse should match to ~1e-4.
    for col in ("age", "avg_glucose_level", "bmi"):
        diff = np.abs(df[col].to_numpy() - df2[col].to_numpy()).max()
        assert diff < 1e-3, f"continuous round-trip failed for '{col}': {diff:.4e}"


def test_opacus_wraps_and_runs_one_step() -> None:
    """PrivacyEngine accepts the masked model and runs one mini DP step.

    Note: full DP-SGD training on CPU is slow (opacus's per-sample grad
    pipeline is GPU-optimised). We exercise *one* gradient step here to
    verify the integration; the production benchmark runner trains for
    real on GPU.
    """
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    df = _make_fake_dataframe(n=128, seed=2)
    schema = infer_stroke_schema(df)
    enc = StrokeEncoder(schema).fit(df)
    x = enc.transform(df)

    model = CDPTabDiffDenoiser(
        schema=schema, embed_dim=8, n_blocks=1, time_dim=16
    )
    sched = DiffusionSchedule.make(
        num_timesteps=20, schedule="cosine", device="cpu"
    )
    trainer = CDPTabDiffTrainer(  # noqa: F841 -- triggers grad-sampler registration
        model=model, schedule=sched, device="cpu",
    )

    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    pe = PrivacyEngine(accountant="rdp")
    model_p, optim_p, loader_p = pe.make_private(
        module=model,
        optimizer=optim,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    (batch,) = next(iter(loader_p))
    t = torch.randint(0, sched.num_timesteps, (batch.shape[0],))
    noise = torch.randn_like(batch)
    x_t = sched.q_sample(batch, t, noise)
    eps_hat = model_p(x_t, t)
    loss = ((eps_hat - noise) ** 2).mean()
    optim_p.zero_grad(set_to_none=True)
    loss.backward()
    optim_p.step()
    assert float(loss.item()) > 0.0


def test_sampling_without_dp() -> None:
    """Reverse process produces a correctly-shaped sample tensor."""
    df = _make_fake_dataframe(n=64, seed=3)
    schema = infer_stroke_schema(df)
    model = CDPTabDiffDenoiser(
        schema=schema, embed_dim=8, n_blocks=1, time_dim=16
    )
    sched = DiffusionSchedule.make(num_timesteps=10, schedule="cosine", device="cpu")
    trainer = CDPTabDiffTrainer(model=model, schedule=sched, device="cpu")
    samples = trainer.generate_samples(num_samples=8, batch_size=4)
    assert samples.shape == (8, model.input_dim)
    df_syn = StrokeEncoder(schema, reference_df=df).fit(df).inverse_transform(samples)
    assert set(df_syn.columns) == set(FEATURE_ORDER)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all() -> Tuple[List[str], List[Tuple[str, Exception]]]:
    tests: List[Tuple[str, Callable[[], None]]] = [
        ("pc_causal_discovery", test_pc_causal_discovery),
        ("learned_denoiser_forward", test_learned_denoiser_forward),
        ("stroke_conditioning_separates_features",
         test_stroke_conditioning_separates_features),
        ("dag_severance", test_dag_severance),
        ("feature_mask_shape_and_zeros", test_feature_mask_shape_and_zeros),
        ("masked_linear_zeros_forbidden_weights",
         test_masked_linear_zeros_forbidden_weights),
        ("denoiser_forward_runs", test_denoiser_forward_runs),
        ("encoder_roundtrip", test_encoder_roundtrip),
        ("opacus_wraps_and_runs_one_step", test_opacus_wraps_and_runs_one_step),
        ("sampling_without_dp", test_sampling_without_dp),
    ]
    passed: List[str] = []
    failed: List[Tuple[str, Exception]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed.append((name, e))
    print(f"\n{len(passed)} passed, {len(failed)} failed.")
    return passed, failed


if __name__ == "__main__":
    _, failed = _run_all()
    sys.exit(1 if failed else 0)
