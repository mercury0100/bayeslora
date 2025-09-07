# bayeslora.py
# Lightweight BayesLoRA for Transformers, compatible with HF/PEFT-style models.
# Author: mercury0100
# License: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Callable, Union
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Optional: PEFT is not required but supported
    from peft import LoraConfig  # type: ignore
except Exception:  # pragma: no cover
    LoraConfig = None  # sentinel


# -----------------------------
# Config
# -----------------------------

@dataclass
class BayesLoraConfig:
    """
    Configuration for BayesLoRA adapters.

    Roughly mirrors PEFT's LoraConfig fields where it makes sense, but adds
    Bayesian-specific knobs.
    """
    r: int = 8
    lora_alpha: int = 16
    target_modules: Sequence[str] = field(default_factory=lambda: ("q_proj","k_proj","v_proj","o_proj"))
    init_scale: float = 1e-4
    prior_keep: float = 0.7                # π0 prior on keep probability
    tau_init: float = 2.0                  # Concrete temperature (will be annealed down)
    tau_final: float = 0.33
    kl_warmup_frac: float = 0.3            # warmup portion to ramp gate-KL
    tau_warmup_frac: float = 0.3           # warmup portion to cool temperature
    kl_max_weight: float = 1.0             # final weight on gate-KL term
    lambda_l2: float = 1e-4                # weight decay on A,B
    tie_across_tokens: bool = True         # one mask per forward() call
    # Advanced: group tying (per-module/per-head). For simplicity we keep one logit per rank by default.
    per_module_shared_mask: bool = False   # if True: share a single gate across all ranks in a module
    # Optional regex to further filter module names (case-sensitive):
    module_name_regex: Optional[str] = None
    # Whether to freeze wrapped base Linear params (LoRA convention)
    freeze_base: bool = True

    @classmethod
    def from_peft(cls, peft_cfg: Any, **overrides) -> "BayesLoraConfig":
        """
        Create from a PEFT LoraConfig-like object.
        Accepts fields: r, lora_alpha, target_modules.
        Other fields can be overridden via kwargs.
        """
        if LoraConfig is not None and isinstance(peft_cfg, LoraConfig):
            r = peft_cfg.r
            lora_alpha = getattr(peft_cfg, "lora_alpha", 16)
            target_modules = getattr(peft_cfg, "target_modules", ("q_proj","k_proj","v_proj","o_proj"))
        else:
            # best-effort duck-typing
            r = getattr(peft_cfg, "r", 8)
            lora_alpha = getattr(peft_cfg, "lora_alpha", 16)
            target_modules = getattr(peft_cfg, "target_modules", ("q_proj","k_proj","v_proj","o_proj"))
        return cls(r=r, lora_alpha=lora_alpha, target_modules=target_modules, **overrides)


# -----------------------------
# Core layer
# -----------------------------

def _bernoulli_kl(pi: torch.Tensor, pi0: Union[float, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    KL( Bern(pi) || Bern(pi0) ), elementwise, then sum over last dimension by caller.
    """
    if not torch.is_tensor(pi0):
        pi0 = torch.as_tensor(pi0, device=pi.device, dtype=pi.dtype)
    pi = pi.clamp(eps, 1 - eps)
    pi0 = pi0.clamp(eps, 1 - eps)
    return pi * (pi.log() - pi0.log()) + (1 - pi) * ((1 - pi).log() - (1 - pi0).log())


class BayesLoraLinear(nn.Module):
    """
    Drop-in replacement for a Linear + LoRA, with learnable rank gates.
    The wrapped base Linear remains deterministic; only the adapter path is stochastic.

    Forward behavior:
      - training: Concrete relaxation (soft gates) for low-variance grads.
      - eval (default): hard Bernoulli gates per forward pass.
      - deterministic mode: use expected keep probs (no sampling).
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: int = 16,
        init_scale: float = 1e-4,
        prior_keep: float = 0.7,
        tau_init: float = 2.0,
        tie_across_tokens: bool = True,
        per_module_shared_mask: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("BayesLoraLinear expects an nn.Linear to wrap.")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, self.r)

        # --- Ensure adapter params/buffers inherit base_linear dtype/device ---
        param_dtype = self.base.weight.dtype
        param_device = self.base.weight.device

        # LoRA parameters A,B (rank r)
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features, dtype=param_dtype, device=param_device))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=param_dtype, device=param_device))
        nn.init.normal_(self.A, std=init_scale)
        nn.init.zeros_(self.B)

        # Gate logits for *drop* probability p_i; keep prob π_i = 1 - sigmoid(alpha_i)
        self.alpha = nn.Parameter(torch.zeros(1 if per_module_shared_mask else self.r, dtype=param_dtype, device=param_device))

        # Buffers (not trainable) — keep dtype/device aligned with params
        self.register_buffer("tau", torch.tensor(float(tau_init), dtype=param_dtype, device=param_device))       # Concrete temperature
        self.register_buffer("prior_keep", torch.tensor(float(prior_keep), dtype=param_dtype, device=param_device))  # π0
        self.tie_across_tokens = bool(tie_across_tokens)
        self.per_module_shared_mask = bool(per_module_shared_mask)

        # Runtime knobs
        self._deterministic: bool = False   # if True, use expected keep (no sampling) in both train/eval
        self._eval_sample: bool = True      # if True, in eval() we sample hard Bernoulli; else use expected
        self._forced_mask: Optional[torch.Tensor] = None  # force a specific keep-mask for this module

    # ---- Gate utilities ----
    def drop_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha)  # p \in (0,1)

    def keep_prob(self) -> torch.Tensor:
        p = self.drop_prob()
        return 1.0 - p  # π = 1 - p

    def expected_keep(self) -> torch.Tensor:
        return self.keep_prob()

    def set_temperature(self, tau: float) -> None:
        self.tau.fill_(float(tau))

    def set_prior_keep(self, pi0: float) -> None:
        self.prior_keep.fill_(float(pi0))

    def set_deterministic(self, enabled: bool = True) -> None:
        self._deterministic = bool(enabled)

    def set_eval_sampling(self, enabled: bool = True) -> None:
        """
        In eval() mode, choose whether to sample hard Bernoulli gates per forward pass.
        If disabled, eval() uses expected keep probs.
        """
        self._eval_sample = bool(enabled)

    @torch.no_grad()
    def set_forced_mask(self, mask: Optional[torch.Tensor]) -> None:
        """
        Force a specific keep-mask for this module (shape [r] or [1] if per_module_shared_mask).
        Pass None to clear.
        """
        if mask is None:
            self._forced_mask = None
            return
        mask = mask.to(device=self.alpha.device, dtype=self.alpha.dtype)
        if self.per_module_shared_mask:
            if mask.numel() != 1:
                raise ValueError("Forced mask for shared-mask module must be scalar (numel==1).")
        else:
            if mask.numel() != self.r:
                raise ValueError(f"Forced mask must have {self.r} elements, got {mask.numel()}.")
        self._forced_mask = mask

    def _sample_concrete_keep(self, shape: torch.Size, training: bool) -> torch.Tensor:
        """
        Returns relaxed keep gates in train, or hard Bernoulli in eval.
        Shape: broadcast to (..., r) unless per_module_shared_mask is True.
        """
        pi = self.expected_keep()  # [...]
        if training:
            # Concrete relaxation on *drop*; convert to keep = 1 - drop
            u = torch.rand(shape + (1 if self.per_module_shared_mask else self.r,),
                           device=self.alpha.device, dtype=self.alpha.dtype)
            drop_relaxed = torch.sigmoid((u.log() - (1 - u).log() + self.alpha) / self.tau)
            keep_relaxed = 1.0 - drop_relaxed
            return keep_relaxed
        else:
            # Hard Bernoulli keep
            keep_hard = torch.bernoulli(pi.expand(shape + (1 if self.per_module_shared_mask else self.r,)))
            return keep_hard

    # ---- Regularizers & diagnostics ----
    def weight_l2(self) -> torch.Tensor:
        return (self.A.pow(2).sum() + self.B.pow(2).sum())

    def gate_kl(self, prior_keep: Optional[float] = None) -> torch.Tensor:
        pi0 = self.prior_keep if prior_keep is None else torch.as_tensor(
            prior_keep, device=self.alpha.device, dtype=self.alpha.dtype
        )
        pi = self.expected_keep()
        kl = _bernoulli_kl(pi.view(-1), pi0).sum()
        return kl

    def effective_rank(self) -> torch.Tensor:
        """
        Sum of keep probabilities; if sharing a single mask, returns π times r.
        """
        pi = self.expected_keep()
        if pi.numel() == 1:
            return pi * self.r
        return pi.sum()

    @torch.no_grad()
    def variance_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cheap scalar uncertainty proxy per sample: tr(Cov) for this adapter.
        Returns shape [...], matching the leading dimensions of x.
        tr(Cov) = sum_i π_i(1-π_i) * z_i^2 * ||b_i||^2  with z = A x
        """
        z = F.linear(x, self.A)  # (..., r)
        pi = self.keep_prob()    # (r,) or (1,)
        s = (pi * (1 - pi)) * (z ** 2)  # broadcast (..., r)
        b_norm2 = self.B.pow(2).sum(dim=0)  # (r,)
        energy = torch.einsum("...r,r->...", s, b_norm2)
        return energy

    # ---- Forward ----
    def forward(self, x: torch.Tensor, sample_gates: Optional[bool] = None) -> torch.Tensor:
        """
        x: [..., in_features]
        sample_gates:
          - True  : stochastic according to train/eval mode (overrides deterministic flag)
          - False : use expected keep probs (deterministic)
          - None  : default behavior (train: relaxed sampling; eval: hard Bernoulli if _eval_sample)
        """
        y = self.base(x)  # deterministic backbone

        # Adapter activation z = A x  => shape [..., r]
        z = F.linear(x, self.A)  # (r, in) @ (..., in) -> (..., r)

        # Forced mask has highest precedence
        if self._forced_mask is not None:
            g = self._forced_mask
            z = z * g
        else:
            # Decide sampling mode
            if self._deterministic or sample_gates is False:
                g = self.expected_keep()  # [...]
                z = z * g
            else:
                # By default, use train/eval behavior
                train_mode = self.training
                if (not train_mode) and (not self._eval_sample):
                    # eval but deterministic eval requested
                    g = self.expected_keep()
                    z = z * g
                else:
                    # sample relaxed in train, hard Bernoulli in eval
                    if self.tie_across_tokens:
                        # share one mask across the whole batch/sequence
                        g = self._sample_concrete_keep(shape=torch.Size([]), training=train_mode)  # [r] or [1]
                        z = z * g
                    else:
                        g = self._sample_concrete_keep(shape=z.shape[:-1], training=train_mode)  # [..., r] or [...,1]
                        z = z * g

        # Project back: adapter output = B @ z, scaled
        y = y + self.scaling * F.linear(z, self.B)
        return y


# -----------------------------
# Model surgery
# -----------------------------

def _matches_any(name: str, substrings: Sequence[str]) -> bool:
    return any(s in name for s in substrings)

def apply_bayeslora(
    model: nn.Module,
    cfg: Union[BayesLoraConfig, Any],
    verbose: bool = True,
) -> nn.Module:
    """
    Replace target nn.Linear modules with BayesLoraLinear wrappers.

    Args:
        model: a HF transformer model (or any nn.Module)
        cfg  : BayesLoraConfig or PEFT LoraConfig (or duck-typed object with r, lora_alpha, target_modules)
        verbose: print a short summary of replacements

    Returns:
        The model (modified in-place) for convenience.
    """
    if not isinstance(cfg, BayesLoraConfig):
        cfg = BayesLoraConfig.from_peft(cfg)  # type: ignore

    name_regex = re.compile(cfg.module_name_regex) if cfg.module_name_regex else None

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if _matches_any(name, cfg.target_modules) and (name_regex is None or name_regex.search(name)):
                # Navigate to parent
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]
                base_linear: nn.Linear = getattr(parent, child_name)

                wrapper = BayesLoraLinear(
                    base_linear=base_linear,
                    r=cfg.r,
                    lora_alpha=cfg.lora_alpha,
                    init_scale=cfg.init_scale,
                    prior_keep=cfg.prior_keep,
                    tau_init=cfg.tau_init,
                    tie_across_tokens=cfg.tie_across_tokens,
                    per_module_shared_mask=cfg.per_module_shared_mask,
                    freeze_base=cfg.freeze_base,
                )
                setattr(parent, child_name, wrapper)
                replaced.append(name)

    if verbose:
        print(f"[BayesLoRA] Wrapped {len(replaced)} Linear modules:", *replaced, sep="\n  - ")
    return model


# -----------------------------
# Training helpers
# -----------------------------

def iter_bayeslora_modules(model: nn.Module) -> Iterable[BayesLoraLinear]:
    for m in model.modules():
        if isinstance(m, BayesLoraLinear):
            yield m

def weight_l2_only(model: nn.Module) -> torch.Tensor:
    p = next(model.parameters())
    total = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    for m in iter_bayeslora_modules(model):
        total = total + m.weight_l2()
    return total

def gate_kl_only(model: nn.Module, prior_keep: Optional[float] = None) -> torch.Tensor:
    p = next(model.parameters())
    total = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    for m in iter_bayeslora_modules(model):
        total = total + m.gate_kl(prior_keep=prior_keep)
    return total

def bayeslora_regularizers(
    model: nn.Module,
    lambda_l2: float,
    lambda_gate: float,
    prior_keep: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (l2_term, gate_kl_term, total_reg = lambda_l2*l2 + lambda_gate*gate_kl)
    """
    l2 = weight_l2_only(model)
    kl = gate_kl_only(model, prior_keep=prior_keep)
    total = lambda_l2 * l2 + lambda_gate * kl
    return l2, kl, total


# -----------------------------
# Schedules
# -----------------------------

def gate_kl_weight_schedule(step: int, total_steps: int, warmup_frac: float = 0.3, max_weight: float = 1.0) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step >= warmup_steps:
        return float(max_weight)
    return float(max_weight) * (step / warmup_steps)

def temperature_schedule(step: int, total_steps: int, start: float = 2.0, end: float = 0.33, warmup_frac: float = 0.3) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step >= warmup_steps:
        return float(end)
    progress = step / warmup_steps
    return float(start + progress * (end - start))

def update_bayeslora_temperatures(model: nn.Module, tau: float) -> None:
    for m in iter_bayeslora_modules(model):
        m.set_temperature(tau)


# -----------------------------
# Inference helpers
# -----------------------------

@torch.no_grad()
def set_expected_gates(model: nn.Module, enabled: bool = True) -> None:
    """
    Force deterministic (expected keep) behavior in all BayesLoRA modules.
    Useful for single-pass deployment (BayesLoRA-Det).
    """
    for m in iter_bayeslora_modules(model):
        m.set_deterministic(enabled)

@torch.no_grad()
def set_sampling_eval(model: nn.Module, enabled: bool = True) -> None:
    """
    When model.eval() is active, choose whether to sample hard Bernoulli gates.
    If disabled, eval uses expected keep probs instead of sampling.
    """
    for m in iter_bayeslora_modules(model):
        m.set_eval_sampling(enabled)

@torch.no_grad()
def mc_predict(
    model: nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    T: int = 8,
    antithetic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run T stochastic adapter passes and return (mean, variance) over outputs.
    - model should be in eval() mode.
    - forward_fn: a zero-arg callable that runs a forward pass and returns a tensor (e.g., logits).
                  It will be invoked T times with different adapter gates.

    If antithetic=True, pairs each sample m with (1 - m) to reduce variance
    using a true paired-mask implementation via forced masks.
    """
    # Ensure eval sampling is enabled unless antithetic overrides it
    set_expected_gates(model, enabled=False)
    set_sampling_eval(model, enabled=True)

    ys: List[torch.Tensor] = []

    if not antithetic:
        for _ in range(T):
            y = forward_fn()
            ys.append(y)
    else:
        modules = list(iter_bayeslora_modules(model))
        half_T = math.ceil(T / 2)
        for _ in range(half_T):
            sampled_masks: List[torch.Tensor] = []
            # Sample a hard keep mask per module (eval -> Bernoulli)
            for m in modules:
                g = m._sample_concrete_keep(shape=torch.Size([]), training=False)
                g = (g >= 0.5).to(g.dtype)  # ensure {0,1}
                sampled_masks.append(g)
                m.set_forced_mask(g)

            y1 = forward_fn()
            ys.append(y1)

            # Complement mask
            for m, g in zip(modules, sampled_masks):
                m.set_forced_mask(1.0 - g)
            y2 = forward_fn()
            ys.append(y2)

            # Clear forced masks
            for m in modules:
                m.set_forced_mask(None)

        ys = ys[:T]

    Y = torch.stack(ys, dim=0)  # [T, ...]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=False)
    return mean, var


# -----------------------------
# Checkpoint helpers
# -----------------------------

def bayeslora_state_dict(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns a nested dict: {module_name: {"A": A, "B": B, "alpha": alpha, "tau": tau, "prior_keep": prior_keep}}
    """
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, BayesLoraLinear):
            state[name] = {
                "A": module.A.detach().cpu(),
                "B": module.B.detach().cpu(),
                "alpha": module.alpha.detach().cpu(),
                "tau": module.tau.detach().cpu(),
                "prior_keep": module.prior_keep.detach().cpu(),
            }
    return state

def load_bayeslora_state_dict(model: nn.Module, state: Dict[str, Dict[str, torch.Tensor]], strict: bool = True) -> None:
    missing: List[str] = []
    for name, params in state.items():
        module = _resolve_module(model, name)
        if not isinstance(module, BayesLoraLinear):
            if strict:
                raise KeyError(f"Module '{name}' is not a BayesLoraLinear in target model.")
            else:
                missing.append(name)
                continue
        with torch.no_grad():
            # Shape checks
            if module.A.shape != params["A"].shape or module.B.shape != params["B"].shape:
                raise ValueError(f"Shape mismatch for '{name}': "
                                 f"A {module.A.shape} vs {tuple(params['A'].shape)}, "
                                 f"B {module.B.shape} vs {tuple(params['B'].shape)}")
            module.A.copy_(params["A"])
            module.B.copy_(params["B"])
            module.alpha.copy_(params["alpha"])
            module.tau.copy_(params["tau"])
            module.prior_keep.copy_(params["prior_keep"])
    if not strict and missing:
        print(f"[BayesLoRA] Skipped {len(missing)} entries not found in model:", *missing, sep="\n  - ")

def _resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    m = model
    for part in dotted_name.split("."):
        m = getattr(m, part)
    return m


# -----------------------------
# Diagnostics
# -----------------------------

@torch.no_grad()
def effective_rank(model: nn.Module) -> float:
    """
    Sum of keep probabilities across all BayesLoRA modules.
    """
    total = 0.0
    for m in iter_bayeslora_modules(model):
        total += float(m.effective_rank().item())
    return total


# -----------------------------
# Minimal usage example (for reference)
# -----------------------------
#
# from transformers import AutoModelForSequenceClassification
# from peft import LoraConfig
# from bayeslora import apply_bayeslora, BayesLoraConfig, bayeslora_regularizers, \
#                       gate_kl_weight_schedule, temperature_schedule, update_bayeslora_temperatures, \
#                       mc_predict, set_expected_gates
#
# model = AutoModelForSequenceClassification.from_pretrained("...")
# peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])
# cfg = BayesLoraConfig.from_peft(peft_cfg, prior_keep=0.7, tau_init=2.0, freeze_base=True)
# apply_bayeslora(model, cfg)
#
# # Training step:
# model.train()
# step, total_steps = 0, 10000
# logits = model(input_ids=..., attention_mask=...).logits
# data_loss = F.cross_entropy(logits, labels)
# # schedules
# lambda_gate = gate_kl_weight_schedule(step, total_steps, warmup_frac=cfg.kl_warmup_frac, max_weight=cfg.kl_max_weight)
# tau = temperature_schedule(step, total_steps, start=cfg.tau_init, end=cfg.tau_final, warmup_frac=cfg.tau_warmup_frac)
# update_bayeslora_temperatures(model, tau)
# # regularizers
# l2, kl, reg = bayeslora_regularizers(model, lambda_l2=cfg.lambda_l2, lambda_gate=lambda_gate, prior_keep=cfg.prior_keep)
# loss = data_loss + reg
# loss.backward()
# optimizer.step()
#
# # Inference (MC):
# model.eval()
# def forward_once():
#     return model(input_ids=..., attention_mask=...).logits
# mean, var = mc_predict(model, forward_once, T=8, antithetic=True)
#
# # Deterministic one-pass (BayesLoRA-Det):
# set_expected_gates(model, enabled=True)
# logits_det = model(input_ids=..., attention_mask=...).logits