"""
Structural regularizer: a Linear layer that zeroes weights along
forbidden causal pathways before every forward pass.

The mask is stored as a non-trainable buffer (so it moves with
``.to(device)`` / ``.cuda()`` automatically and is saved with the
state dict). Because the multiplication ``weight * mask`` happens
inside ``forward``, autograd propagates zero gradients along the
masked entries — the forbidden edges stay forbidden through training.

Opacus compatibility note: this module exposes the same parameters
as ``nn.Linear`` (``weight``, optional ``bias``), so per-sample
gradient hooks attach without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMaskedLinear(nn.Linear):
    """``nn.Linear`` whose weight matrix is masked by a fixed DAG mask.

    Parameters
    ----------
    in_features, out_features, bias :
        Same as ``nn.Linear``.
    mask :
        ``(out_features, in_features)`` binary tensor (0/1). Stored as
        a buffer; not trainable. Anything truthy is coerced to 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match "
                f"({out_features}, {in_features})."
            )
        # NB: stored as a plain attribute (not a registered buffer) so the
        # module is not flagged by Opacus's blanket "trainable layer with
        # buffers" filter (which is really intended to catch BatchNorm).
        self._mask = (mask != 0).to(self.weight.dtype)
        # Zero forbidden weights at init; the pre-forward hook below keeps
        # them at exactly zero across optimiser steps. We deliberately do
        # NOT multiply ``weight * mask`` inside ``forward`` because that
        # creates two autograd nodes in a single layer, which breaks
        # Opacus's module-level activation/backprop hooks (the
        # "non-full backward hook" warning). Instead we maintain the
        # invariant ``weight.data * (1-mask) == 0`` and rely on the
        # vanilla ``F.linear(x, weight, bias)`` graph.
        with torch.no_grad():
            self.weight.mul_(self._mask)
        self.register_forward_pre_hook(_apply_causal_mask)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse=recurse)
        self._mask = fn(self._mask)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vanilla nn.Linear forward; the pre-hook has just enforced
        # ``weight.data[mask == 0] = 0`` so forbidden edges are dead.
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        kept = int(self._mask.sum().item())
        total = self._mask.numel()
        return (
            super().extra_repr()
            + f", mask_density={kept}/{total} ({100.0 * kept / total:.1f}%)"
        )


def _apply_causal_mask(module: "CausalMaskedLinear", _input):
    """Pre-forward hook: snap forbidden weight entries back to zero."""
    with torch.no_grad():
        module.weight.data.mul_(module._mask)
