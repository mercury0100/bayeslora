# bayeslora.py
# MIT-like minimal license: use at your own risk :)

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Defaults (Recipe B-ish)
# -----------------------------

@dataclass(init=False)
class BayesLoRAConfig:
    # LoRA core
    rank: int = 64
    lora_alpha: int = 128
    freeze_base: bool = True

    # Variational / ARD
    local_reparam: bool = True
    kl_scale: float = 1e-3
    kl_freeze_epochs: int = 25
    beta_warmup_epochs: int = 10
    sample_warmup_epochs: int = 15
    logalpha_thresh: float = 3.5  # prune if logalpha > thresh

    # Optim / misc
    lr: float = 5e-4
    epochs: int = 50

    # Targets (suffix or exact attr names). Includes DistilBERT & common names.
    target_modules: Tuple[str, ...] = (
        # DistilBERT
        "q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2",
        # Common
        "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj",
        "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
        "fc1", "fc2"
    )

    # Extras commonly passed around by scripts/repos
    lora_dropout: float = 0.0
    bias: str = "none"
    fan_in_fan_out: bool = False

    # Classifier head control
    train_classifier_head: bool = True
    classifier_head_names: Tuple[str, ...] = ("classifier", "pre_classifier")

    # capture unknown kwargs so cfg(**anything) doesn't explode
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        # apply defaults
        fields = dataclasses.fields(self.__class__)
        for f in fields:
            if f.default is not dataclasses.MISSING:
                setattr(self, f.name, f.default)
            elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:
                setattr(self, f.name, f.default_factory())  # type: ignore
        self.extra_kwargs = {}

        # alias map to absorb variations across repos
        alias_map = {
            "r": "rank",
            "lora_rank": "rank",
            "alpha": "lora_alpha",
            "dropout": "lora_dropout",
            "head_names": "classifier_head_names",
        }

        valid = {f.name for f in fields}
        for k, v in list(kwargs.items()):
            key = alias_map.get(k, k)
            if key in valid:
                setattr(self, key, v)
            else:
                # accept but ignore unknowns
                self.extra_kwargs[key] = v


def kl_weight_for_epoch(epoch: int, kl_freeze_epochs: int, beta_warmup_epochs: int) -> float:
    """
    Cosine warmup after a freeze phase:
    - 0 until kl_freeze_epochs
    - smooth rise to 1 over beta_warmup_epochs
    """
    if epoch < kl_freeze_epochs:
        return 0.0
    if beta_warmup_epochs <= 0:
        return 1.0
    t = (epoch - kl_freeze_epochs + 1) / beta_warmup_epochs
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def sample_prob_for_epoch(epoch: int, sample_warmup_epochs: int) -> float:
    """Linear ramp posterior sampling probability."""
    if sample_warmup_epochs <= 0:
        return 0.0
    return min(1.0, (epoch + 1) / sample_warmup_epochs)


# -----------------------------
# Vanilla LoRA
# -----------------------------

class LoRALinear(nn.Module):
    """
    Drop-in linear with LoRA (no bias change). Adds low-rank ΔW = s * A @ B.
    """
    def __init__(self, base_linear: nn.Linear, rank: int, lora_alpha: int,
                 lora_dropout: float = 0.0, **_ignored):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.scaling = float(lora_alpha) / max(1, self.rank)

        # wrap base params (do not clone)
        self.weight = base_linear.weight
        self.bias = base_linear.bias

        # LoRA factors
        self.A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        # Optional dropout on LoRA path (like PEFT)
        self.lora_dropout_p = float(lora_dropout)
        self.dropout = nn.Dropout(self.lora_dropout_p) if self.lora_dropout_p > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.A)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        x_lora = self.dropout(x)
        delta = F.linear(x_lora, self.B @ self.A, None) * self.scaling
        return y + delta


# -----------------------------
# Variational LoRA (SVD-ARD)
# -----------------------------

def molchanov_kl_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Molchanov et al. (2017) KL approx for log-uniform prior:
    KL(logα) ≈ k1*sigmoid(k2 + k3*logα) - 0.5*softplus(-logα) - k1
    """
    k1, k2, k3 = 0.63576, 1.87320, 1.48695
    return (k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * F.softplus(-log_alpha)
            - k1)


class VariationalLoRALinear(nn.Module):
    """
    Variational LoRA with ARD per rank (A columns & B rows).
    A ~ N(A_mu, α_A * A_mu^2), α_A = exp(logalpha_A)
    B ~ N(B_mu, α_B * B_mu^2), α_B = exp(logalpha_B)

    Pruning: mask ranks where logalpha > threshold for either A or B.
    """
    def __init__(self,
                 base_linear: nn.Linear,
                 rank: int,
                 lora_alpha: int,
                 kl_scale: float = 1e-3,
                 logalpha_thresh: float = 3.5,
                 local_reparam: bool = True,
                 lora_dropout: float = 0.0,
                 **_ignored):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.scaling = float(lora_alpha) / max(1, self.rank)
        self.kl_scale = float(kl_scale)
        self.logalpha_thresh = float(logalpha_thresh)
        self.local_reparam = bool(local_reparam)

        # base params (shared)
        self.weight = base_linear.weight
        self.bias = base_linear.bias

        # posterior means
        self.A_mu = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.B_mu = nn.Parameter(torch.zeros(self.out_features, self.rank))

        # log-alpha gates (ARD)
        self.logalpha_A = nn.Parameter(torch.full((self.rank,), -5.0))
        self.logalpha_B = nn.Parameter(torch.full((self.rank,), -5.0))

        # prune masks
        self.register_buffer("keep_mask_A", torch.ones(self.rank, dtype=torch.bool))
        self.register_buffer("keep_mask_B", torch.ones(self.rank, dtype=torch.bool))

        # optional dropout on LoRA path
        self.lora_dropout_p = float(lora_dropout)
        self.dropout = nn.Dropout(self.lora_dropout_p) if self.lora_dropout_p > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.A_mu)
        nn.init.kaiming_uniform_(self.B_mu, a=math.sqrt(5))

    @torch.no_grad()
    def prune_(self, threshold: Optional[float] = None):
        """Mask ranks where logalpha > threshold (either A or B)."""
        thr = float(self.logalpha_thresh if threshold is None else threshold)
        keep_A = self.logalpha_A <= thr
        keep_B = self.logalpha_B <= thr
        keep = keep_A & keep_B
        self.keep_mask_A = keep.clone()
        self.keep_mask_B = keep.clone()

    def effective_rank(self) -> int:
        return int(self.keep_mask_A.sum().item())

    def _sample_AB(self, sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sample:
            A = self.A_mu * self.keep_mask_A[:, None]
            B = self.B_mu * self.keep_mask_B[None, :]
            return A, B

        eps_A = torch.randn_like(self.A_mu)
        eps_B = torch.randn_like(self.B_mu)

        alpha_A = torch.exp(self.logalpha_A).clamp_min(1e-9)
        alpha_B = torch.exp(self.logalpha_B).clamp_min(1e-9)

        sigma_A = (alpha_A[:, None].sqrt() * self.A_mu.abs()).clamp_min(1e-12)
        sigma_B = (alpha_B[None, :].sqrt() * self.B_mu.abs()).clamp_min(1e-12)

        A = (self.A_mu + sigma_A * eps_A) * self.keep_mask_A[:, None]
        B = (self.B_mu + sigma_B * eps_B) * self.keep_mask_B[None, :]
        return A, B

    def forward(self, x: torch.Tensor, sample: bool = False, sample_prob: float = 0.0):
        if (not sample) and (sample_prob > 0.0):
            sample = bool(torch.rand(()) < sample_prob)

        y = F.linear(x, self.weight, self.bias)
        x_lora = self.dropout(x)
        A, B = self._sample_AB(sample)
        delta = F.linear(x_lora, B @ A, None) * self.scaling
        return y + delta

    def kl(self) -> torch.Tensor:
        kl_A_r = molchanov_kl_log_alpha(self.logalpha_A)  # [r]
        kl_B_r = molchanov_kl_log_alpha(self.logalpha_B)  # [r]
        wA = float(self.in_features)
        wB = float(self.out_features)
        kl_total = (kl_A_r * wA + kl_B_r * wB).sum()
        return self.kl_scale * kl_total


# -----------------------------
# Utilities to inject adapters
# -----------------------------

def _replace_linear_with(module: nn.Module,
                         path: List[str],
                         ctor) -> None:
    """
    Replace a nested nn.Linear at 'path' in module with ctor(old_linear).
    """
    curr = module
    for p in path[:-1]:
        curr = getattr(curr, p)
    leaf_name = path[-1]
    old = getattr(curr, leaf_name)
    new = ctor(old)
    setattr(curr, leaf_name, new)


def _iter_named_modules_by_suffix(model: nn.Module,
                                  suffixes: Iterable[str]) -> Iterable[Tuple[str, nn.Module]]:
    suf = tuple(suffixes)
    for n, m in model.named_modules():
        if n.endswith(suf):
            yield n, m


def apply_lora(model: nn.Module,
               cfg: BayesLoRAConfig,
               *,
               variational: bool = True) -> List[str]:
    """
    Wrap target nn.Linear modules in LoRA (vanilla or variational).
    Returns list of replaced module names.
    """
    replaced: List[str] = []

    def builder(old_linear: nn.Linear):
        if variational:
            return VariationalLoRALinear(
                base_linear=old_linear,
                rank=cfg.rank,
                lora_alpha=cfg.lora_alpha,
                kl_scale=_effective_kl_scale(cfg),
                logalpha_thresh=cfg.logalpha_thresh,
                local_reparam=cfg.local_reparam,
            )
        else:
            return LoRALinear(
                base_linear=old_linear,
                rank=cfg.rank,
                lora_alpha=cfg.lora_alpha,
            )

    # Find target modules
    named = dict(model.named_modules())
    target_names = set()

    for name in cfg.target_modules:
        if name in named:
            target_names.add(name)
    if not target_names:  # suffix match
        for name, _ in _iter_named_modules_by_suffix(model, cfg.target_modules):
            target_names.add(name)

    for name in sorted(target_names, key=lambda s: (len(s.split(".")), s)):
        path = name.split(".")
        parent = model
        for p in path[:-1]:
            parent = getattr(parent, p)
        leaf = getattr(parent, path[-1])
        if isinstance(leaf, nn.Linear):
            _replace_linear_with(model, path, builder)
            replaced.append(name)

    # === Freeze/unfreeze logic ===
    if cfg.freeze_base:
        # freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze LoRA adapter params
        for m in get_lora_modules(model):
            for n, p in m.named_parameters(recurse=True):
                if any(key in n for key in ["A_mu", "B_mu", "logalpha_A", "logalpha_B"]):
                    p.requires_grad = True

        # unfreeze classifier head if asked
        if getattr(cfg, "train_classifier_head", True):
            head_names = getattr(cfg, "classifier_head_names",
                                 ("classifier", "pre_classifier"))
            for n, p in model.named_parameters():
                if any(h in n for h in head_names):
                    p.requires_grad = True

    return replaced


def get_lora_modules(model: nn.Module) -> List[nn.Module]:
    mods: List[nn.Module] = []
    for _, m in model.named_modules():
        if isinstance(m, (LoRALinear, VariationalLoRALinear)):
            mods.append(m)
    return mods


def compute_total_kl(model: nn.Module) -> torch.Tensor:
    dev = None
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    total = torch.tensor(0.0, device=dev)
    for m in get_lora_modules(model):
        if isinstance(m, VariationalLoRALinear):
            total = total + m.kl()
    return total


@torch.no_grad()
def prune_all_lora(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Prune (mask) all variational LoRA ranks with their module thresholds.
    Returns dict {module_name: {"before": r_before, "after": r_after}}
    """
    report: Dict[str, Dict[str, int]] = {}
    for name, m in model.named_modules():
        if isinstance(m, VariationalLoRALinear):
            before = m.effective_rank()
            m.prune_()
            after = m.effective_rank()
            report[name] = {"before": before, "after": after}
    return report


@torch.no_grad()
def prune_vlora_by_logalpha(model: nn.Module, threshold: Optional[float] = None) -> Dict[str, Dict[str, int]]:
    """
    Alias expected by some scripts. If threshold is given, override per-module.
    """
    report: Dict[str, Dict[str, int]] = {}
    for name, m in model.named_modules():
        if isinstance(m, VariationalLoRALinear):
            before = m.effective_rank()
            m.prune_(threshold=threshold)
            after = m.effective_rank()
            report[name] = {"before": before, "after": after}
    return report


# -----------------------------
# Helpers: KL auto-scaling
# -----------------------------

def _effective_kl_scale(cfg: BayesLoRAConfig) -> float:
    """
    Keep total KL pressure roughly stable across ranks and lora_alpha.
    """
    base = cfg.kl_scale
    size_correction = (64 / max(1, cfg.rank)) * (cfg.lora_alpha / max(1, cfg.rank)) ** 0.5
    return float(base * size_correction)


# -----------------------------
# (Optional) Tiny smoke test
# -----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, 10, bias=False),
    )

    cfg = BayesLoRAConfig(
        rank=64, lora_alpha=128,
        freeze_base=True,
        local_reparam=True,
        kl_scale=1e-3,
        kl_freeze_epochs=25,
        beta_warmup_epochs=10,
        sample_warmup_epochs=15,
        logalpha_thresh=3.5,
        lora_dropout=0.05,
        # verify aliases/unknowns don't crash:
        r=64, alpha=128, fan_in_fan_out=False, some_unknown_flag=True,
        train_classifier_head=True, head_names=("head",),
        target_modules=("0", "2"),
    )

    replaced = apply_lora(model, cfg, variational=True)
    print("Replaced:", replaced)

    x = torch.randn(8, 32)
    y = model(x)
    print("Output shape:", y.shape)

    kl = compute_total_kl(model)
    print("KL:", float(kl))

    rep = prune_vlora_by_logalpha(model, threshold=3.5)
    print("Prune report:", rep)