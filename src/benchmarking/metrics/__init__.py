"""Modular evaluation metrics for synthetic data benchmarking."""

from .fairness import compute_fairness
from .fidelity import compute_fidelity
from .privacy import compute_dcr, compute_mia, compute_mia_success_rate
from .utility import compute_tstr

__all__ = [
    "compute_tstr",
    "compute_fairness",
    "compute_dcr",
    "compute_mia",
    "compute_mia_success_rate",
    "compute_fidelity",
]
