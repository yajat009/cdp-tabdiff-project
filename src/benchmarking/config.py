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

# Stroke class conditioning (fixes label-feature decoupling at sampling)
USE_STROKE_CONDITIONING = True      # broadcast stroke label embedding to all layers
USE_STROKE_TRAIN_INPAINTING = True  # keep stroke block aligned to label during training
DIFFUSION_CFG_SCALE = 2.0           # classifier-free guidance scale at sampling (>1 amplifies)

# Plan A — class-conditional adaptive DP noise (toggleable)
USE_ADAPTIVE_DP_NOISE = True
ADAPTIVE_DP_MINORITY_NOISE_SCALE = 0.5   # less noise on stroke=1 gradients
ADAPTIVE_DP_MAJORITY_NOISE_SCALE = 1.0   # base noise on stroke=0 gradients

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
    "ctgan": {"n_iter": N_ITER, "batch_size": 500, "pac": 1},
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
