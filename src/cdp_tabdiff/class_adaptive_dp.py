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
    minority_grad_weight: float = 1.0,
    majority_grad_weight: float = 1.0,
) -> DPOptimizer:
    """
    Attach class-conditional noise hooks to an existing Opacus ``DPOptimizer``.

    Two independent knobs per class:

    * ``*_noise_scale`` controls the *privacy* of that class. Each class's
      clipped gradient sum is noised at
      ``noise_multiplier * max_grad_norm * noise_scale``. The class's
      Rényi-DP guarantee is governed solely by this scale (the reported
      ``epsilon`` from ``make_private_with_epsilon`` assumes the majority
      scale; reducing the minority scale weakens *its* guarantee — logged).
    * ``*_grad_weight`` controls the *optimization* influence of that class.
      Critically, on a 5%-positive dataset under DP-SGD the loss-level
      reweighting trick is cancelled by per-sample gradient clipping (every
      row, up-weighted or not, is clipped to ``max_grad_norm``), so the only
      way to restore minority influence is to reweight the already-clipped,
      already-noised class contributions here. Because the weight multiplies
      the class's signal *and* its privacy-calibrated noise by the same
      factor, it does not change that class's privacy guarantee.

    Normalisation by the expected batch size is intentionally left to
    Opacus's ``scale_grad`` (called after ``add_noise`` in ``pre_step``),
    which divides ``p.grad`` by ``expected_batch_size`` for mean reduction.

    Returns the same optimizer instance (mutated in place).
    """
    # Sentinel default (all-majority) so a missing ``set_stroke_labels`` call
    # never silently degrades into the no-noise fallback path.
    optimizer._cdp_stroke_labels = None  # type: ignore[attr-defined]
    optimizer._cdp_minority_noise_scale = float(minority_noise_scale)  # type: ignore[attr-defined]
    optimizer._cdp_majority_noise_scale = float(majority_noise_scale)  # type: ignore[attr-defined]
    optimizer._cdp_minority_grad_weight = float(minority_grad_weight)  # type: ignore[attr-defined]
    optimizer._cdp_majority_grad_weight = float(majority_grad_weight)  # type: ignore[attr-defined]

    def set_stroke_labels(labels: Optional[torch.Tensor]) -> None:
        optimizer._cdp_stroke_labels = labels  # type: ignore[attr-defined]

    def clip_and_accumulate_classwise() -> None:
        n = len(optimizer.grad_samples[0])
        labels = optimizer._cdp_stroke_labels  # type: ignore[attr-defined]
        # Guard the first-step / unset race: treat unknown labels as all
        # majority so every parameter still receives correctly-scaled noise
        # (never the unnoised bypass path).
        if labels is None or len(labels) != n:
            labels = torch.zeros(n, dtype=torch.long)
        labels = labels.to(optimizer.grad_samples[0].device)

        if n == 0:
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
        minority_weight = optimizer._cdp_minority_grad_weight  # type: ignore[attr-defined]
        majority_weight = optimizer._cdp_majority_grad_weight  # type: ignore[attr-defined]

        for p in optimizer.params:
            grad_sample = optimizer._get_flat_grad_sample(p)

            # grad_total has the parameter's own shape. NB: do NOT pre-sum
            # over the batch with contract("i,i...") first — that collapses
            # the per-sample axis and makes per-class indexing impossible.
            grad_total = torch.zeros_like(grad_sample[0])
            for class_val, noise_scale, grad_weight in (
                (0, majority_scale, majority_weight),
                (1, minority_scale, minority_weight),
            ):
                mask = labels == class_val
                if not bool(mask.any()):
                    continue
                # Sum of *clipped per-sample* gradients within this class:
                #   group_sum = sum_{i in class} clip_factor_i * grad_sample_i
                group_sum = contract(
                    "i,i...->...",
                    per_sample_clip_factor[mask],
                    grad_sample[mask],
                )
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
                # grad_weight scales signal and privacy-calibrated noise
                # together -> optimization-only knob, privacy unchanged.
                grad_total = grad_total + grad_weight * (group_sum + noise)

            p.grad = grad_total.view_as(p)
            _mark_as_processed(p.grad_sample)

    def add_noise_bypass() -> None:
        """Noise already applied class-wise inside ``clip_and_accumulate_classwise``."""
        return None

    optimizer.set_stroke_labels = set_stroke_labels  # type: ignore[attr-defined]
    optimizer.clip_and_accumulate = clip_and_accumulate_classwise  # type: ignore[method-assign]
    optimizer.add_noise = add_noise_bypass  # type: ignore[method-assign]
    return optimizer
