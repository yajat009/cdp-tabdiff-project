"""Shared diffusion training + sampling pipeline for TabDDPM variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from benchmarking.config import (
    ADAPTIVE_DP_MAJORITY_GRAD_WEIGHT,
    ADAPTIVE_DP_MAJORITY_NOISE_SCALE,
    ADAPTIVE_DP_MINORITY_GRAD_WEIGHT,
    ADAPTIVE_DP_MINORITY_NOISE_SCALE,
    DATASET_NAME,
    DIFFUSION_CLIP_ZSCORE,
    DIFFUSION_P_UNCOND,
    DIFFUSION_CFG_SCALE,
    MERGE_LEARNED_WITH_EXPERT,
    PC_ALPHA,
    PC_INDEP_TEST,
    STROKE_DIM_LOSS_WEIGHT,
    STROKE_SAMPLE_LOSS_WEIGHT,
    CONTINUOUS_DIM_LOSS_WEIGHT,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_RATE,
    USE_ADAPTIVE_DP_NOISE,
    USE_LEARNED_DAG,
    USE_OVERSAMPLING,
    USE_REBALANCING,
    USE_STROKE_CONDITIONING,
    USE_STROKE_LOSS_REWEIGHTING,
    USE_STROKE_TRAIN_INPAINTING,
)
from benchmarking.data import load_stroke_data
from benchmarking.imbalance_utils import (
    compute_positive_rate,
    finalize_synthetic_output,
    prepare_synthesizer_training_data,
)
from cdp_tabdiff import (
    CDPTabDiffDenoiser,
    CDPTabDiffTrainer,
    DiffusionSchedule,
    StrokeEncoder,
    discover_learned_dag,
    infer_stroke_schema,
    log_dag_data_support,
)


@dataclass
class DiffusionRunConfig:
    model_name: str
    mask_type: str = "causal"
    dp_enabled: bool = True
    epochs: int = 50
    batch_size: int = 256
    embed_dim: int = 32
    n_blocks: int = 4
    timesteps: int = 1000
    schedule_kind: str = "cosine"
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    lr: float = 1e-3
    stroke_dim_loss_weight: float = STROKE_DIM_LOSS_WEIGHT
    stroke_sample_loss_weight: float = STROKE_SAMPLE_LOSS_WEIGHT
    continuous_dim_loss_weight: float = CONTINUOUS_DIM_LOSS_WEIGHT
    use_stroke_loss_reweighting: bool = USE_STROKE_LOSS_REWEIGHTING
    use_stroke_conditioning: bool = USE_STROKE_CONDITIONING
    use_stroke_train_inpainting: bool = USE_STROKE_TRAIN_INPAINTING
    cfg_guidance_scale: float = DIFFUSION_CFG_SCALE
    use_adaptive_dp_noise: bool = USE_ADAPTIVE_DP_NOISE
    adaptive_dp_minority_noise_scale: float = ADAPTIVE_DP_MINORITY_NOISE_SCALE
    adaptive_dp_majority_noise_scale: float = ADAPTIVE_DP_MAJORITY_NOISE_SCALE
    adaptive_dp_minority_grad_weight: float = ADAPTIVE_DP_MINORITY_GRAD_WEIGHT
    adaptive_dp_majority_grad_weight: float = ADAPTIVE_DP_MAJORITY_GRAD_WEIGHT
    p_uncond: float = DIFFUSION_P_UNCOND
    target_positive_rate: float = TARGET_POSITIVE_RATE
    use_oversampling: bool = USE_OVERSAMPLING
    oversample_strategy: float = 0.2
    use_rebalancing: bool = USE_REBALANCING
    random_mask_seed: int = RANDOM_STATE
    use_learned_dag: bool = USE_LEARNED_DAG
    pc_alpha: float = PC_ALPHA
    pc_indep_test: str = PC_INDEP_TEST
    merge_learned_with_expert: bool = MERGE_LEARNED_WITH_EXPERT
    sample_size: Optional[int] = None
    output_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(
            f"[GPU] Using {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.device_count()} device(s))"
        )
    else:
        print("[GPU] CUDA not available — training on CPU")
    return dev


def run_diffusion_benchmark(cfg: DiffusionRunConfig) -> tuple[pd.DataFrame, Dict]:
    """
    Train a TabDDPM variant, sample synthetic rows, optionally rebalance,
    and persist CSV + checkpoint artifacts.
    """
    _seed_everything(RANDOM_STATE)
    device = _device()

    print(f"\n=== Diffusion benchmark: {cfg.model_name.upper()} ===")
    print(
        f"mask={cfg.mask_type} dp={cfg.dp_enabled} epochs={cfg.epochs} "
        f"eps={cfg.target_epsilon} p_uncond={cfg.p_uncond} "
        f"use_oversampling={cfg.use_oversampling} "
        f"use_rebalancing={cfg.use_rebalancing} "
        f"stroke_cond={cfg.use_stroke_conditioning} "
        f"cfg_scale={cfg.cfg_guidance_scale}"
    )

    full_df = (
        load_stroke_data(sample_size=cfg.sample_size)
        if cfg.sample_size
        else load_stroke_data()
    )
    train_df, _ = prepare_synthesizer_training_data(
        full_df,
        use_oversampling=cfg.use_oversampling,
        sampling_strategy=cfg.oversample_strategy,
        random_state=RANDOM_STATE,
    )
    print(f"Training fold shape: {train_df.shape}")
    print(
        f"Train positive rate: "
        f"{compute_positive_rate(train_df, TARGET_COLUMN):.4f}"
    )

    log_dag_data_support(train_df)
    schema = infer_stroke_schema(train_df)
    encoder = StrokeEncoder(
        schema=schema,
        clip_zscore=DIFFUSION_CLIP_ZSCORE,
        reference_df=train_df,
    ).fit(train_df)
    x_train = encoder.transform(train_df)
    print(f"Encoded tensor shape: {tuple(x_train.shape)}")

    mask_type = cfg.mask_type
    learned_adj = None
    if cfg.use_learned_dag or mask_type == "learned":
        mask_type = "learned"
        learned_adj = discover_learned_dag(
            train_df,
            alpha=cfg.pc_alpha,
            indep_test=cfg.pc_indep_test,
            random_state=RANDOM_STATE,
            merge_with_expert=cfg.merge_learned_with_expert,
            log_fn=print,
        )

    model = CDPTabDiffDenoiser(
        schema=schema,
        embed_dim=cfg.embed_dim,
        n_blocks=cfg.n_blocks,
        time_dim=128,
        dropout=0.0,
        mask_type=mask_type,
        random_mask_seed=cfg.random_mask_seed,
        learned_adj=learned_adj,
    )
    if mask_type in {"causal", "learned"}:
        model.log_mask_statistics()

    schedule = DiffusionSchedule.make(
        num_timesteps=cfg.timesteps,
        schedule=cfg.schedule_kind,
        device=device,
    )

    loss_weights = torch.ones(encoder.input_dim, dtype=torch.float32)
    stroke_slice = model.feature_slice("stroke")
    print(
        f"[Plan C] stroke loss reweighting={cfg.use_stroke_loss_reweighting} "
        f"dim_weight={cfg.stroke_dim_loss_weight} "
        f"sample_weight={cfg.stroke_sample_loss_weight}"
    )
    if cfg.use_adaptive_dp_noise and cfg.dp_enabled:
        print(
            f"[Plan A] adaptive DP noise: minority_scale="
            f"{cfg.adaptive_dp_minority_noise_scale} majority_scale="
            f"{cfg.adaptive_dp_majority_noise_scale}"
        )

    trainer = CDPTabDiffTrainer(
        model=model,
        schedule=schedule,
        device=device,
        lr=cfg.lr,
        max_grad_norm=cfg.max_grad_norm,
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta,
        sample_loss_weights=None,
        p_uncond=cfg.p_uncond,
        stroke_slice=stroke_slice,
        target_positive_rate=cfg.target_positive_rate,
        use_stroke_conditioning=cfg.use_stroke_conditioning,
        use_stroke_train_inpainting=cfg.use_stroke_train_inpainting,
        cfg_guidance_scale=cfg.cfg_guidance_scale,
        use_stroke_loss_reweighting=cfg.use_stroke_loss_reweighting,
        stroke_dim_loss_weight=cfg.stroke_dim_loss_weight,
        stroke_sample_loss_weight=cfg.stroke_sample_loss_weight,
        continuous_dim_loss_weight=cfg.continuous_dim_loss_weight,
        use_adaptive_dp_noise=cfg.use_adaptive_dp_noise and cfg.dp_enabled,
        adaptive_dp_minority_noise_scale=cfg.adaptive_dp_minority_noise_scale,
        adaptive_dp_majority_noise_scale=cfg.adaptive_dp_majority_noise_scale,
        adaptive_dp_minority_grad_weight=cfg.adaptive_dp_minority_grad_weight,
        adaptive_dp_majority_grad_weight=cfg.adaptive_dp_majority_grad_weight,
    )

    print(f"Training (dp_enabled={cfg.dp_enabled})...")
    trainer.train(
        x_train=x_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        dp_enabled=cfg.dp_enabled,
    )

    eps = trainer.get_epsilon() if cfg.dp_enabled else float("inf")
    delta = cfg.target_delta if cfg.dp_enabled else float("nan")
    print(f"[DP] final epsilon = {eps} at delta = {delta}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = RESULTS_DIR / f"{cfg.model_name}_checkpoint.pt"
    trainer.save_checkpoint(cfg.checkpoint_path)
    print(f"[checkpoint] saved weights to {cfg.checkpoint_path}")

    n_samples = len(full_df)
    print(
        f"Generating {n_samples:,} synthetic rows with "
        f"target rate={cfg.target_positive_rate:.4f}..."
    )
    samples = trainer.generate_samples(
        num_samples=n_samples,
        batch_size=512,
        target_positive_rate=cfg.target_positive_rate,
        clip_zscore=DIFFUSION_CLIP_ZSCORE,
    )
    synthetic_raw = encoder.inverse_transform(samples)
    _log_decode_sanity(train_df, synthetic_raw)

    synthetic, rates = finalize_synthetic_output(
        synthetic_raw,
        cfg.model_name,
        use_rebalancing=cfg.use_rebalancing,
        target_rate=cfg.target_positive_rate,
        random_state=RANDOM_STATE,
        assert_rebalanced=cfg.use_rebalancing,
    )

    if cfg.output_path is None:
        cfg.output_path = RESULTS_DIR / f"synthetic_{DATASET_NAME}_{cfg.model_name}.csv"
    synthetic.to_csv(cfg.output_path, index=False)
    print(f"Saved synthetic data to {cfg.output_path}")

    train_cache = RESULTS_DIR / "real_train_reference.csv"
    if not train_cache.exists():
        full_df.to_csv(train_cache, index=False)
        print(f"Cached real training reference to {train_cache}")

    meta = {
        "model": cfg.model_name,
        "mask_type": mask_type,
        "use_learned_dag": int(cfg.use_learned_dag or mask_type == "learned"),
        "merge_learned_with_expert": int(cfg.merge_learned_with_expert),
        "dp_enabled": int(cfg.dp_enabled),
        "epsilon": eps,
        "delta": delta,
        "max_grad_norm": cfg.max_grad_norm,
        "stroke_dim_loss_weight": cfg.stroke_dim_loss_weight,
        "stroke_sample_loss_weight": cfg.stroke_sample_loss_weight,
        "use_adaptive_dp_noise": int(cfg.use_adaptive_dp_noise and cfg.dp_enabled),
        "adaptive_dp_minority_noise_scale": cfg.adaptive_dp_minority_noise_scale,
        "epochs": cfg.epochs,
        "timesteps": cfg.timesteps,
        "syn_positive_rate_before": rates["rate_before"],
        "syn_positive_rate_after": rates["rate_after"],
        "synthetic_path": str(cfg.output_path),
        "checkpoint_path": str(cfg.checkpoint_path),
    }
    return synthetic, meta


def _log_decode_sanity(train_df: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    """Log whether decoded continuous columns stay within training support."""
    for col in ("age", "avg_glucose_level", "bmi"):
        if col not in synthetic.columns:
            continue
        lo, hi = float(train_df[col].min()), float(train_df[col].max())
        syn_lo, syn_hi = float(synthetic[col].min()), float(synthetic[col].max())
        oob = int(((synthetic[col] < lo) | (synthetic[col] > hi)).sum())
        print(
            f"[decode] {col}: train=[{lo:.2f}, {hi:.2f}] "
            f"syn=[{syn_lo:.2f}, {syn_hi:.2f}] out_of_range={oob}"
        )
