"""Shared configuration for the synthetic-data benchmarking suite."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_CSV = DATA_DIR / "healthcare-dataset-stroke-data.csv"

DATASET_NAME = "stroke"
TARGET_COLUMN = "stroke"
SENSITIVE_COLUMN = "gender"
DROP_COLUMNS = ["id"]

RANDOM_STATE = 42
# Stroke dataset only has ~5k rows; keep full data by default.
SAMPLE_SIZE = 5_000
N_ITER = 300

# --- Class-imbalance controls (toggleable) ---
USE_OVERSAMPLING = True
USE_REBALANCING = True
TARGET_POSITIVE_RATE = 0.0488  # real train positive rate (195/4000)
OVERSAMPLE_STRATEGY = 0.2      # minority = 20% of majority after ROS
REBALANCE_RATE_MIN = 0.03
REBALANCE_RATE_MAX = 0.08

# Diffusion-specific training knobs
DIFFUSION_P_UNCOND = 0.10          # classifier-free guidance dropout prob.
DIFFUSION_CLIP_ZSCORE = 4.0        # clip continuous dims during sampling/decode
CDP_TABDIFF_EPSILON = 8.0          # primary DP budget
CDP_TABDIFF_EPSILON_SENSITIVITY = (10.0, 15.0)  # optional sensitivity runs

# Plan C — minority-aware diffusion loss re-weighting (DP + imbalance)
USE_STROKE_LOSS_REWEIGHTING = True
STROKE_DIM_LOSS_WEIGHT = 15.0       # up-weight MSE on stroke one-hot dimensions
STROKE_SAMPLE_LOSS_WEIGHT = 15.0    # per-row multiplier for stroke=1 in weighted MSE

# Per-dimension MSE weight on the 3 continuous columns (age, avg_glucose_level,
# bmi). Only 3 of 24 encoded dims are continuous, so an unweighted DDPM MSE
# lets the ~21 one-hot categorical dims dominate the gradient; the continuous
# score then never trains and the reverse process diverges to the clip bounds.
# Up-weighting the continuous dims (~21/3 to equalise their loss contribution)
# fixes the collapse. Set to 1.0 to disable.
# Up-weighting the continuous dims helps the non-DP model but HURTS the DP
# model (the DP-noised continuous gradient can't exploit the extra budget and
# the loss is swamped), so it is disabled (1.0) by default. The lever is kept
# for experimentation. The continuous mode-collapse fix that actually matters
# is the x0-clamped reverse step in CDPTabDiffTrainer.generate_samples.
CONTINUOUS_DIM_LOSS_WEIGHT = 1.0

# Stroke class conditioning (fixes label-feature decoupling at sampling)
USE_STROKE_CONDITIONING = True      # broadcast stroke label embedding to all layers
USE_STROKE_TRAIN_INPAINTING = True  # keep stroke block aligned to label during training
DIFFUSION_CFG_SCALE = 2.0           # classifier-free guidance scale at sampling (>1 amplifies). NOTE: cfg>1 over-saturates the continuous dims somewhat (biases glucose/bmi high, age low), BUT it is essential for stroke class separation — cfg=1.0 collapses TSTR AUROC to ~0.5. Keep >=2.0.

# Plan A — class-conditional adaptive DP noise (toggleable)
USE_ADAPTIVE_DP_NOISE = True
ADAPTIVE_DP_MINORITY_NOISE_SCALE = 0.5   # less noise on stroke=1 gradients (privacy knob)
ADAPTIVE_DP_MAJORITY_NOISE_SCALE = 1.0   # base noise on stroke=0 gradients (privacy knob)
# Optimization knob: up-weight the (clipped, noised) minority-class gradient
# contribution so it is not out-voted by the majority sum. Loss-level
# reweighting is cancelled by per-sample clipping under DP-SGD, so this is the
# effective lever for minority signal. Scales signal and privacy-calibrated
# noise together => does not change the per-class privacy guarantee.
ADAPTIVE_DP_MINORITY_GRAD_WEIGHT = 6.0
ADAPTIVE_DP_MAJORITY_GRAD_WEIGHT = 1.0
# Exclude the 2x time_dim stroke conditioning embedding from DP noise so the
# two class codes don't collapse together under the noise multiplier.
FREEZE_STROKE_COND_EMBED = True

# Backwards-compatible alias
DIFFUSION_STROKE_LOSS_WEIGHT = STROKE_DIM_LOSS_WEIGHT

# Causal discovery (PC algorithm via causal-learn)
USE_LEARNED_DAG = False
PC_ALPHA = 0.05
PC_INDEP_TEST = "auto"  # auto | fisherz | chisq_binned
MERGE_LEARNED_WITH_EXPERT = True  # union of learned + expert edges before fairness cut

SYNTHCITY_MODELS = ["ctgan", "tvae", "adsgan", "dpgan", "pategan"]
SDV_MODELS = ["sdv_ctgan", "sdv_tvae", "sdv_gaussian_copula", "sdv_copula_gan"]
NOVEL_MODELS = ["cdp_tabdiff"]
# Ablation variants emitted by ``run_ablations.py``. ``cdp_tabdiff`` itself
# is already in NOVEL_MODELS; we keep it there as the canonical name and
# do not duplicate it in this list.
ABLATION_MODELS = [
    "base_tabddpm",
    "fair_tabdiff",
    "dp_tabddpm_dense",
    "dp_tabddpm_random",
]
MODELS = tuple(SYNTHCITY_MODELS + SDV_MODELS + NOVEL_MODELS + ABLATION_MODELS)

# Default plugin hyperparameters (override per script if needed).
# Keys are model names (with prefix for SDV variants).
PLUGIN_KWARGS = {
    # --- SynthCity plugins ---
    # ``target_column`` is set on GenericDataLoader for conditional training.
    # NOTE: this synthcity build's CTGAN plugin does not accept ``pac``.
    "ctgan": {"n_iter": N_ITER, "batch_size": 500},
    "tvae": {"n_iter": N_ITER, "batch_size": 500},
    "adsgan": {"n_iter": N_ITER, "batch_size": 500},
    "dpgan": {"n_iter": N_ITER, "batch_size": 500, "epsilon": 1.0},
    "pategan": {
        "n_iter": N_ITER,
        "batch_size": 500,
        "epsilon": 1.0,
        "n_teachers": 10,
        "random_state": RANDOM_STATE,
    },
    # --- SDV synthesizers (epochs map roughly to N_ITER for neural models) ---
    "sdv_ctgan": {
        "epochs": N_ITER,
        "batch_size": 500,
        "enforce_min_max_values": True,
        "discriminator_steps": 1,
    },
    "sdv_tvae": {
        "epochs": N_ITER,
        "batch_size": 500,
        "enforce_min_max_values": True,
    },
    "sdv_gaussian_copula": {"enforce_min_max_values": True},
    "sdv_copula_gan": {
        "epochs": N_ITER,
        "batch_size": 500,
        "enforce_min_max_values": True,
        "discriminator_steps": 1,
    },
}
