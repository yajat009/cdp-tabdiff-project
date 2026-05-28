"""
Class-adaptive DP-SGD noise for imbalanced tabular diffusion training.

Standard Opacus adds the same Gaussian noise to the aggregated clipped
gradient of an entire batch. When the positive class is ~5% of rows, its
per-sample gradient signal is drowned out. This module patches a wrapped
:class:`~opacus.optimizers.DPOptimizer` so that clipped gradients are
summed *within* each stroke class, noised with class-specific scales, and
then combined.

Privacy note: the reported ``(epsilon, delta)`` budget from
``make_private_with_epsilon`` assumes the base ``noise_multiplier`` on the
*majority* class. Reducing noise on the minority class weakens the
minority's individual guarantee; both scales are logged at training start.
"""

from __future__ import annotations

from typing import Optional

import torch
from opacus.optimizers.optimizer import DPOptimizer, _generate_noise, _mark_as_processed
from opt_einsum.contract import contract


def patch_class_adaptive_dp(
    optimizer: DPOptimizer,
    *,
    minority_noise_scale: float = 0.5,
    majority_noise_scale: float = 1.0,
) -> DPOptimizer:
    """
    Attach class-conditional noise hooks to an existing Opacus ``DPOptimizer``.

    Returns the same optimizer instance (mutated in place).
    """
    optimizer._cdp_stroke_labels = None  # type: ignore[attr-defined]
    optimizer._cdp_minority_noise_scale = float(minority_noise_scale)  # type: ignore[attr-defined]
    optimizer._cdp_majority_noise_scale = float(majority_noise_scale)  # type: ignore[attr-defined]

    def set_stroke_labels(labels: Optional[torch.Tensor]) -> None:
        optimizer._cdp_stroke_labels = labels  # type: ignore[attr-defined]

    def clip_and_accumulate_classwise() -> None:
        labels = optimizer._cdp_stroke_labels  # type: ignore[attr-defined]
        if labels is None:
            return DPOptimizer.clip_and_accumulate(optimizer)

        if len(optimizer.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1)
                for g in optimizer.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                optimizer.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        minority_scale = optimizer._cdp_minority_noise_scale  # type: ignore[attr-defined]
        majority_scale = optimizer._cdp_majority_noise_scale  # type: ignore[attr-defined]

        for p in optimizer.params:
            grad_sample = optimizer._get_flat_grad_sample(p)
            clipped = contract("i,i...", per_sample_clip_factor, grad_sample)

            grad_total = torch.zeros_like(clipped[0])
            for class_val, noise_scale in (
                (0, majority_scale),
                (1, minority_scale),
            ):
                mask = labels == class_val
                if not bool(mask.any()):
                    continue
                group_sum = clipped[mask].sum(dim=0)
                noise_std = (
                    optimizer.noise_multiplier
                    * optimizer.max_grad_norm
                    * noise_scale
                )
                noise = _generate_noise(
                    std=noise_std,
                    reference=group_sum,
                    generator=optimizer.generator,
                    secure_mode=optimizer.secure_mode,
                )
                grad_total = grad_total + group_sum + noise

            p.grad = grad_total.view_as(p)
            _mark_as_processed(p.grad_sample)

    def add_noise_bypass() -> None:
        """Noise already applied class-wise inside ``clip_and_accumulate_classwise``."""
        return None

    optimizer.set_stroke_labels = set_stroke_labels  # type: ignore[attr-defined]
    optimizer.clip_and_accumulate = clip_and_accumulate_classwise  # type: ignore[method-assign]
    optimizer.add_noise = add_noise_bypass  # type: ignore[method-assign]
    return optimizer
