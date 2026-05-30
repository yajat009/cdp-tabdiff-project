"""
DP-SGD diffusion trainer for CDP-TabDiff.

The trainer

* runs a standard DDPM training objective (MSE between predicted noise and
  the true Gaussian noise added at timestep ``t``),
* wraps the model + optimizer + dataloader in Opacus's ``PrivacyEngine``
  so every parameter update is differentially private,
* tracks the running :math:`(\\varepsilon, \\delta)` budget via RDP
  accounting, and
* exposes a ``generate_samples(num_samples)`` method that runs the DDPM
  ancestral reverse-process to draw fresh synthetic rows.

Opacus quirks that we handle:

1. ``CausalMaskedLinear.forward`` uses ``weight * mask``. Opacus's default
   per-sample gradient hook for ``nn.Linear`` would compute the gradient
   w.r.t. the *effective* (masked) weight and ignore the mask, which both
   pollutes the gradient-norm clipping budget and adds DP noise to
   forbidden weight entries. We register a custom grad-sampler so the
   per-sample gradient is multiplied by ``mask`` before clipping.
2. Only ``LayerNorm`` is used inside the model (BatchNorm is unsupported
   by DP-SGD because it couples samples).

PyTorch 2.2.2 + Opacus 1.4/1.5 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .class_adaptive_dp import patch_class_adaptive_dp
from .mask import CausalMaskedLinear
from .model import CDPTabDiffDenoiser


# ---------------------------------------------------------------------------
# Opacus integration: custom per-sample gradient for CausalMaskedLinear
# ---------------------------------------------------------------------------
_OPACUS_REGISTERED = False


def _register_opacus_grad_sampler() -> None:
    """Register per-sample grad and module-validator entries for
    :class:`CausalMaskedLinear`.

    Opacus has *two* gates:

    1. ``GradSampleModule.GRAD_SAMPLERS`` — dict of class -> per-sample
       gradient function. This determines whether the layer can be
       wrapped during training.
    2. ``ModuleValidator`` — separate allow-list used by
       ``PrivacyEngine.make_private`` to up-front reject "unsupported"
       layers before any per-sample machinery runs.

    Both need our class registered. Idempotent.
    """
    global _OPACUS_REGISTERED
    if _OPACUS_REGISTERED:
        return
    try:
        from opacus.grad_sample import register_grad_sampler
        from opacus.grad_sample.linear import compute_linear_grad_sample
        from opacus.validators import register_module_validator
        from opacus.validators.utils import (
            register_module_fixer,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Opacus is required for CDPTabDiffTrainer. Install with "
            "`pip install opacus`."
        ) from e

    @register_grad_sampler(CausalMaskedLinear)
    def _causal_masked_linear_grad_sampler(
        layer: CausalMaskedLinear,
        activations: List[torch.Tensor],
        backprops: torch.Tensor,
    ):
        # Reuse opacus's well-tested nn.Linear per-sample grad, then
        # multiply weight grads by the mask so DP clipping and noise
        # are applied to the *correct* (non-forbidden) entries only.
        per_sample_grads = compute_linear_grad_sample(layer, activations, backprops)
        if layer.weight in per_sample_grads:
            per_sample_grads[layer.weight] = (
                per_sample_grads[layer.weight] * layer.mask
            )
        return per_sample_grads

    # Module-validator side: declare the layer safe (no fixup needed).
    @register_module_validator(CausalMaskedLinear)
    def _validate_causal_masked_linear(module):  # noqa: ARG001
        return []

    @register_module_fixer(CausalMaskedLinear)
    def _fix_causal_masked_linear(module):
        return module

    _OPACUS_REGISTERED = True


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    """DDPM noise schedule precomputed on a chosen device/dtype."""

    num_timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @staticmethod
    def make(
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "DiffusionSchedule":
        if schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=dtype
            )
        elif schedule == "cosine":
            # Nichol & Dhariwal (2021), s=0.008
            s = 0.008
            ts = torch.arange(num_timesteps + 1, dtype=dtype) / num_timesteps
            f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = torch.clamp(1.0 - alphas_bar[1:] / alphas_bar[:-1], 1e-8, 0.999)
        else:
            raise ValueError(f"unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod)
        # posterior_variance_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=dtype), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        to = lambda x: x.to(device=device, dtype=dtype)
        return DiffusionSchedule(
            num_timesteps=num_timesteps,
            betas=to(betas),
            alphas=to(alphas),
            alphas_cumprod=to(alphas_cumprod),
            sqrt_alphas_cumprod=to(sqrt_alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=to(sqrt_one_minus),
            posterior_variance=to(posterior_variance),
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward (noising) process: draw ``x_t`` given ``x_0`` and ``t``."""
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return a * x0 + b * noise


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    last_loss: float = float("nan")
    epsilon: float = 0.0
    delta: float = 0.0


class CDPTabDiffTrainer:
    """End-to-end DP-SGD trainer for the CDP-TabDiff denoiser."""

    def __init__(
        self,
        model: CDPTabDiffDenoiser,
        schedule: Optional[DiffusionSchedule] = None,
        device: torch.device | str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        sample_loss_weights: Optional[torch.Tensor] = None,
        p_uncond: float = 0.10,
        stroke_slice: Optional[slice] = None,
        target_positive_rate: float = 0.0488,
        clip_zscore: float = 4.0,
        use_stroke_conditioning: bool = True,
        use_stroke_train_inpainting: bool = True,
        cfg_guidance_scale: float = 2.0,
        use_stroke_loss_reweighting: bool = True,
        stroke_dim_loss_weight: float = 15.0,
        stroke_sample_loss_weight: float = 15.0,
        continuous_dim_loss_weight: float = 1.0,
        use_adaptive_dp_noise: bool = True,
        adaptive_dp_minority_noise_scale: float = 0.5,
        adaptive_dp_majority_noise_scale: float = 1.0,
        adaptive_dp_minority_grad_weight: float = 6.0,
        adaptive_dp_majority_grad_weight: float = 1.0,
        freeze_stroke_cond_embed: bool = True,
    ) -> None:
        _register_opacus_grad_sampler()

        self.device = torch.device(device)
        self.model = model.to(self.device)
        # Cache attributes of the underlying denoiser; after Opacus wraps it
        # ``self.model`` becomes a ``GradSampleModule`` which only proxies
        # ``forward`` and not custom attributes (``input_dim`` etc.).
        self._raw_model: CDPTabDiffDenoiser = model
        self.input_dim: int = model.input_dim
        self.schedule = schedule or DiffusionSchedule.make(device=self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        # Optional per-dimension loss reweighting (e.g. up-weight the stroke
        # column to compensate for severe class imbalance).
        if sample_loss_weights is not None:
            sample_loss_weights = sample_loss_weights.to(self.device)
        self.sample_loss_weights = sample_loss_weights
        self.p_uncond = p_uncond
        self.stroke_slice = stroke_slice or model.feature_slice("stroke")
        self.target_positive_rate = target_positive_rate
        self.clip_zscore = clip_zscore
        self.use_stroke_conditioning = use_stroke_conditioning
        self.use_stroke_train_inpainting = use_stroke_train_inpainting
        self.cfg_guidance_scale = cfg_guidance_scale
        self.use_stroke_loss_reweighting = use_stroke_loss_reweighting
        self.stroke_dim_loss_weight = stroke_dim_loss_weight
        self.stroke_sample_loss_weight = stroke_sample_loss_weight
        self.use_adaptive_dp_noise = use_adaptive_dp_noise
        self.adaptive_dp_minority_noise_scale = adaptive_dp_minority_noise_scale
        self.adaptive_dp_majority_noise_scale = adaptive_dp_majority_noise_scale
        self.adaptive_dp_minority_grad_weight = adaptive_dp_minority_grad_weight
        self.adaptive_dp_majority_grad_weight = adaptive_dp_majority_grad_weight
        self.freeze_stroke_cond_embed = freeze_stroke_cond_embed
        self._continuous_slices = [
            model.feature_slice(spec.name)
            for spec in model.schema
            if not spec.is_categorical
        ]
        # Per-dimension MSE weighting: up-weight the (few) continuous dims so
        # they are not drowned out by the many one-hot categorical dims. Only
        # built here when an explicit per-dim vector wasn't already provided.
        self.continuous_dim_loss_weight = continuous_dim_loss_weight
        if self.sample_loss_weights is None and continuous_dim_loss_weight != 1.0:
            dim_w = torch.ones(self.input_dim, dtype=torch.float32, device=self.device)
            for sl in self._continuous_slices:
                dim_w[sl] = continuous_dim_loss_weight
            self.sample_loss_weights = dim_w
        self._rng = np.random.default_rng(42)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.privacy_engine = None  # set by .train()
        self.state = TrainState()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _stroke_labels_from_x0(self, x0: torch.Tensor) -> torch.Tensor:
        """Binary stroke labels from the one-hot block in encoded rows."""
        block = x0[:, self.stroke_slice]
        return block.argmax(dim=1).long()

    def _cfg_dropout_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.p_uncond <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) < self.p_uncond

    def _apply_cfg_dropout(
        self,
        x0: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Zero the stroke block for unconditional (CFG) training rows."""
        if not bool(drop_mask.any()):
            return x0
        x0 = x0.clone()
        x0[drop_mask, self.stroke_slice] = 0.0
        return x0

    def _model_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stroke_labels: torch.Tensor,
        stroke_cond_drop: Optional[torch.Tensor] = None,
        *,
        module: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Route through Opacus wrapper during training, raw model otherwise."""
        mod = module or (self.model if self.model.training else self._raw_model)
        if self.use_stroke_conditioning:
            return mod(
                x_t,
                t,
                stroke_labels=stroke_labels,
                stroke_cond_drop=stroke_cond_drop,
            )
        return mod(x_t, t)

    def _predict_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        stroke_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise with optional classifier-free guidance at sampling."""
        scale = self.cfg_guidance_scale
        if self.use_stroke_conditioning and scale > 1.0:
            drop_none = torch.zeros(
                x.shape[0], dtype=torch.bool, device=x.device
            )
            drop_all = torch.ones(
                x.shape[0], dtype=torch.bool, device=x.device
            )
            eps_cond = self._model_forward(
                x, t, stroke_labels, stroke_cond_drop=drop_none, module=self._raw_model
            )
            eps_uncond = self._model_forward(
                x, t, stroke_labels, stroke_cond_drop=drop_all, module=self._raw_model
            )
            return eps_uncond + scale * (eps_cond - eps_uncond)
        return self._model_forward(
            x, t, stroke_labels, module=self._raw_model
        )

    def _stroke_onehot_from_labels(
        self, labels: torch.Tensor, width: int, device: torch.device
    ) -> torch.Tensor:
        """Map binary stroke labels to one-hot blocks of width ``width``."""
        oh = torch.zeros(labels.shape[0], width, device=device)
        oh[torch.arange(labels.shape[0], device=device), labels.long()] = 1.0
        return oh

    def _build_class_labels(self, num_samples: int) -> np.ndarray:
        """Sample binary labels in proportion to the real positive rate."""
        n_pos = max(1, int(round(num_samples * self.target_positive_rate)))
        n_pos = min(n_pos, num_samples)
        labels = np.zeros(num_samples, dtype=np.int64)
        labels[:n_pos] = 1
        self._rng.shuffle(labels)
        return labels

    def _condition_stroke_slice(
        self,
        x: torch.Tensor,
        t_int: int,
        x0_stroke: torch.Tensor,
        stroke_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Inpaint the stroke block so it matches the fixed class label at ``t``."""
        if t_int <= 0:
            x[:, self.stroke_slice] = x0_stroke
            return x
        alpha_bar = self.schedule.alphas_cumprod[t_int]
        a = torch.sqrt(alpha_bar)
        b = torch.sqrt(1.0 - alpha_bar)
        x[:, self.stroke_slice] = a * x0_stroke + b * stroke_noise
        return x

    def _condition_stroke_slice_batched(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0_stroke: torch.Tensor,
        stroke_noise: torch.Tensor,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched stroke inpainting for per-row diffusion timesteps."""
        if keep_mask is not None and not bool(keep_mask.any()):
            return x
        if keep_mask is None:
            idx = torch.arange(x.shape[0], device=x.device)
        else:
            idx = keep_mask.nonzero(as_tuple=True)[0]
        alpha_bar = self.schedule.alphas_cumprod[t[idx]].view(-1, 1)
        a = torch.sqrt(alpha_bar)
        b = torch.sqrt(1.0 - alpha_bar)
        x[idx, self.stroke_slice] = (
            a * x0_stroke[idx] + b * stroke_noise[idx]
        )
        return x

    def _diffusion_loss(
        self,
        x0: torch.Tensor,
        stroke_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stroke_labels is None:
            stroke_labels = self._stroke_labels_from_x0(x0)
        B = x0.shape[0]
        drop_mask = self._cfg_dropout_mask(B, x0.device)
        x0 = self._apply_cfg_dropout(x0, drop_mask)
        t = torch.randint(
            low=0,
            high=self.schedule.num_timesteps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        if self.use_stroke_train_inpainting:
            stroke_width = self.stroke_slice.stop - self.stroke_slice.start
            x0_stroke = self._stroke_onehot_from_labels(
                stroke_labels, stroke_width, self.device
            )
            keep = ~drop_mask if self.use_stroke_conditioning else None
            x_t = self._condition_stroke_slice_batched(
                x_t,
                t,
                x0_stroke,
                noise[:, self.stroke_slice],
                keep_mask=keep,
            )
        eps_hat = self._model_forward(
            x_t,
            t,
            stroke_labels=stroke_labels,
            stroke_cond_drop=drop_mask if self.use_stroke_conditioning else None,
        )

        # Weighted DP-MSE. Two independent reweightings, applied together:
        #   * per-dimension weights up-weight the continuous columns so their
        #     score actually trains (they are only ~12% of the encoded dims);
        #   * per-sample weights amplify stroke=1 rows before Opacus clipping.
        sq_error = (eps_hat - noise) ** 2
        if self.sample_loss_weights is not None:
            sq_error = sq_error * self.sample_loss_weights  # (D,) broadcast over (B, D)
        if self.use_stroke_loss_reweighting:
            weights = torch.where(
                stroke_labels == 1,
                torch.full((B,), self.stroke_sample_loss_weight, device=self.device),
                torch.ones(B, device=self.device),
            )
            return torch.mean(weights.view(B, 1) * sq_error)
        return sq_error.mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        x_train: torch.Tensor,
        epochs: int,
        batch_size: int = 256,
        log_every: int = 50,
        log_fn: Callable[[str], None] = print,
        dp_enabled: bool = True,
    ) -> TrainState:
        """
        Fit the denoiser on the encoded tabular tensor ``x_train`` of
        shape ``(N, model.input_dim)``.

        If ``dp_enabled=True`` (the default), the model, optimizer and
        dataloader are wrapped in Opacus's ``PrivacyEngine`` and the
        ``epochs`` count is used to size the noise multiplier so the
        requested ``(target_epsilon, target_delta)`` is met after
        exactly that many passes through the data.

        If ``dp_enabled=False`` we run vanilla PyTorch SGD/AdamW with
        no gradient clipping or noise. Useful for ablations measuring
        the cost of DP-SGD vs. the unconstrained utility ceiling.
        """
        if x_train.ndim != 2 or x_train.shape[1] != self.input_dim:
            raise ValueError(
                f"x_train must be (N, {self.input_dim}); got "
                f"{tuple(x_train.shape)}."
            )

        x_train = x_train.float().to(self.device)
        stroke_labels_all = self._stroke_labels_from_x0(x_train)
        pos_rate = float((stroke_labels_all == 1).float().mean().item())
        log_fn(
            f"[imbalance] train stroke rate={pos_rate:.4f} "
            f"loss_reweight={self.use_stroke_loss_reweighting} "
            f"dim_w={self.stroke_dim_loss_weight} sample_w={self.stroke_sample_loss_weight}"
        )

        dataset = TensorDataset(x_train, stroke_labels_all)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # DP noise should apply only to score-network weights, not the
        # conditioning pathways. The time-embedding MLP and the stroke
        # class-conditioning embedding are excluded from DP-SGD: they are
        # tiny label/timestep lookups, and leaving the 2x time_dim stroke
        # embedding inside DP lets the heavy noise multiplier (~1.3 at
        # eps=8) collapse the stroke=0 and stroke=1 vectors toward each
        # other, which destroys class conditioning at sampling time. The
        # downstream score network (which actually maps these codes to
        # class-conditional features) remains fully private.
        if dp_enabled:
            for param in self.model.time_embed.parameters():
                param.requires_grad = False
            if self.freeze_stroke_cond_embed and hasattr(
                self.model, "stroke_cond_embed"
            ):
                for param in self.model.stroke_cond_embed.parameters():
                    param.requires_grad = False

        if dp_enabled:
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.model, self.optimizer, loader = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=loader,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    epochs=epochs,
                    max_grad_norm=self.max_grad_norm,
                )
            )
            log_fn(
                f"[DP] noise_multiplier={self.optimizer.noise_multiplier:.4f} "
                f"max_grad_norm={self.max_grad_norm} "
                f"target=({self.target_epsilon}, {self.target_delta})"
            )
            if self.use_adaptive_dp_noise:
                patch_class_adaptive_dp(
                    self.optimizer,
                    minority_noise_scale=self.adaptive_dp_minority_noise_scale,
                    majority_noise_scale=self.adaptive_dp_majority_noise_scale,
                    minority_grad_weight=self.adaptive_dp_minority_grad_weight,
                    majority_grad_weight=self.adaptive_dp_majority_grad_weight,
                )
                log_fn(
                    "[DP] class-adaptive noise enabled: "
                    f"minority_scale={self.adaptive_dp_minority_noise_scale} "
                    f"majority_scale={self.adaptive_dp_majority_noise_scale} "
                    f"minority_grad_weight={self.adaptive_dp_minority_grad_weight} "
                    f"majority_grad_weight={self.adaptive_dp_majority_grad_weight} "
                    "(epsilon budget assumes majority scale)"
                )
        else:
            self.privacy_engine = None
            log_fn("[DP] disabled — running standard (non-private) training")

        self.model.train()
        for epoch in range(epochs):
            for batch, batch_labels in loader:
                batch = batch.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                if hasattr(self.optimizer, "set_stroke_labels"):
                    self.optimizer.set_stroke_labels(batch_labels)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._diffusion_loss(batch, stroke_labels=batch_labels)
                loss.backward()
                self.optimizer.step()

                self.state.step += 1
                self.state.last_loss = float(loss.detach().item())
                if log_every and self.state.step % log_every == 0:
                    if self.privacy_engine is not None:
                        eps = self.privacy_engine.get_epsilon(
                            delta=self.target_delta
                        )
                        log_fn(
                            f"epoch {epoch + 1}/{epochs} step {self.state.step}: "
                            f"loss={self.state.last_loss:.4f} eps={eps:.3f}"
                        )
                    else:
                        log_fn(
                            f"epoch {epoch + 1}/{epochs} step {self.state.step}: "
                            f"loss={self.state.last_loss:.4f}"
                        )

            self.state.epoch = epoch + 1
            if self.privacy_engine is not None:
                self.state.epsilon = self.privacy_engine.get_epsilon(
                    delta=self.target_delta
                )
                self.state.delta = self.target_delta
                log_fn(
                    f"[end epoch {epoch + 1}] loss={self.state.last_loss:.4f} "
                    f"({self.state.epsilon:.3f}, {self.state.delta})"
                )
            else:
                self.state.epsilon = float("inf")
                self.state.delta = float("nan")
                log_fn(
                    f"[end epoch {epoch + 1}] loss={self.state.last_loss:.4f}"
                )

        # Whether DP-wrapped or not, sync to the raw model so sampling
        # and checkpointing always read from a single, stable handle.
        self._sync_weights_from_wrapped()
        return self.state

    def _sync_weights_from_wrapped(self, log_fn: Callable[[str], None] = print) -> None:
        """Copy trained params from the Opacus wrapper onto the raw model.

        Opacus prefixes module keys with ``_module.`` and the prefix-stripping
        behaviour differs across versions; a silent ``strict=False`` load can
        leave the raw model at its random init, so ``generate_samples`` would
        run on an untrained network (a classic "loss drops but samples are
        garbage" collapse). We therefore log any key mismatch and assert that
        at least one parameter actually changed.
        """
        # Reference a *trainable* parameter; frozen conditioning params
        # (time_embed / stroke_cond_embed) legitimately never change and
        # would trigger a false "unchanged" warning.
        trainable = {
            name
            for name, p in self._raw_model.named_parameters()
            if p.requires_grad
        }
        raw_sd = self._raw_model.state_dict()
        ref_name = next((k for k in raw_sd if k in trainable), None)
        if ref_name is None:
            ref_name = next(iter(raw_sd))
        ref_before = raw_sd[ref_name].detach().clone()
        try:
            wrapped = getattr(self.model, "_module", self.model)
            missing, unexpected = self._raw_model.load_state_dict(
                wrapped.state_dict(), strict=False
            )
            if missing:
                log_fn(f"[warn] weight sync missing keys: {list(missing)}")
            if unexpected:
                log_fn(f"[warn] weight sync unexpected keys: {list(unexpected)}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not sync weights from wrapped model: {exc}")
            return
        ref_after = self._raw_model.state_dict()[ref_name]
        delta = float((ref_after - ref_before).abs().max().item())
        if delta == 0.0:
            log_fn(
                f"[warn] weight sync: '{ref_name}' unchanged after training "
                "(possible failed sync — sampling may use untrained weights)."
            )
        else:
            log_fn(f"[sync] raw model updated (max|Δ '{ref_name}'|={delta:.4g}).")

    def save_checkpoint(self, path) -> None:
        """Persist the trained denoiser weights to ``path``."""
        torch.save(
            {
                "model_state_dict": self._raw_model.state_dict(),
                "epsilon": self.state.epsilon,
                "delta": self.state.delta,
            },
            str(path),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _clip_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp continuous feature blocks to a sane z-score range."""
        if self.clip_zscore <= 0:
            return x
        out = x.clone()
        for sl in self._continuous_slices:
            out[:, sl] = torch.clamp(out[:, sl], -self.clip_zscore, self.clip_zscore)
        return out

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        batch_size: int = 512,
        clip_x0: bool = True,
        target_positive_rate: Optional[float] = None,
        clip_zscore: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Draw ``num_samples`` synthetic rows via the DDPM reverse process.

        Class labels for ``stroke`` are fixed to match ``target_positive_rate``
        (default: trainer's configured rate) rather than the model's learned
        marginal, using inpainting on the stroke one-hot block.
        """
        if target_positive_rate is not None:
            old_rate = self.target_positive_rate
            self.target_positive_rate = target_positive_rate
        else:
            old_rate = None
        if clip_zscore is not None:
            old_clip = self.clip_zscore
            self.clip_zscore = clip_zscore
        else:
            old_clip = None

        self._raw_model.eval()
        self.model.eval()
        sched = self.schedule
        stroke_sl = self.stroke_slice
        stroke_width = stroke_sl.stop - stroke_sl.start
        class_labels = self._build_class_labels(num_samples)

        chunks: List[torch.Tensor] = []
        remaining = num_samples
        offset = 0
        while remaining > 0:
            B = min(batch_size, remaining)
            labels = torch.from_numpy(class_labels[offset : offset + B]).to(
                self.device
            )
            x0_stroke = self._stroke_onehot_from_labels(
                labels, stroke_width, self.device
            )
            stroke_noise = torch.randn(B, stroke_width, device=self.device)

            x = torch.randn(B, self.input_dim, device=self.device)
            x = self._condition_stroke_slice(
                x, sched.num_timesteps - 1, x0_stroke, stroke_noise
            )

            for t_int in reversed(range(sched.num_timesteps)):
                t = torch.full((B,), t_int, device=self.device, dtype=torch.long)
                eps_hat = self._predict_eps(x, t, labels)
                beta_t = sched.betas[t_int]
                alpha_t = sched.alphas[t_int]
                alpha_bar_t = sched.alphas_cumprod[t_int]
                # Predict x0 from eps, clamp its continuous dims to a sane
                # z-range, then build the DDPM posterior mean from the
                # *clamped x0*. Clamping the latent instead (the old
                # behaviour) let imperfect-score errors accumulate over the
                # 1000-step reverse chain and drove the unbounded continuous
                # dims out to the clip boundary -> continuous mode collapse.
                x0_hat = (
                    x - torch.sqrt(1.0 - alpha_bar_t) * eps_hat
                ) / torch.sqrt(alpha_bar_t)
                if clip_x0:
                    x0_hat = self._clip_continuous(x0_hat)
                alpha_bar_prev = (
                    sched.alphas_cumprod[t_int - 1]
                    if t_int > 0
                    else torch.ones((), device=self.device)
                )
                coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
                coef_xt = (
                    torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                )
                mean = coef_x0 * x0_hat + coef_xt * x
                if t_int > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(sched.posterior_variance[t_int])
                    x = mean + sigma * noise
                else:
                    x = mean
                x = self._condition_stroke_slice(
                    x, max(t_int - 1, 0), x0_stroke, stroke_noise
                )

            x = self._clip_continuous(x)
            chunks.append(x.cpu())
            remaining -= B
            offset += B

        if old_rate is not None:
            self.target_positive_rate = old_rate
        if old_clip is not None:
            self.clip_zscore = old_clip
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Privacy bookkeeping
    # ------------------------------------------------------------------
    def get_epsilon(self) -> float:
        if self.privacy_engine is None:
            return 0.0
        return float(
            self.privacy_engine.get_epsilon(delta=self.target_delta)
        )
