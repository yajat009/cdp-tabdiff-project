"""Path bootstrap so scripts can be run directly from the project root."""

import sys
import warnings
from pathlib import Path

# Synthcity's early-stopping metrics call sklearn clustering scores on mixed
# dtypes and spam thousands of identical UserWarnings to stderr during training.
warnings.filterwarnings(
    "ignore",
    message="Clustering metrics expects discrete values*",
    category=UserWarning,
    module="sklearn.metrics.cluster._supervised",
)
warnings.filterwarnings(
    "ignore",
    message="Disabling PyTorch because PyTorch >= 2.4 is required*",
)
warnings.filterwarnings(
    "ignore",
    message="PyTorch was not found. Models won't be available*",
)

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
