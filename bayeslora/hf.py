# bayeslora/hf.py
# QLoRA-safe BayesLoRA wrappers:
# - DO NOT copy base weights (works with bitsandbytes Linear4bit/8bit & nn.Linear)
# - Keep adapter params & ARD math in FP32 to avoid NaNs; cast to activation dtype in forward
# - Expose helpers used by your benchmark (sampling toggle, KL step, prune, rank/param counts)

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Molchanov et al. 2017 constants for KL =====
_K1, _K2, _K3 = 0.63576, 1.87320, 1.48695
_EPS = 1e-8

def _kl_from_mu_logs2(mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    # FP32 math for stability
    mu32 = mu.float()
    log_sigma2_32 = log_sigma2.float()
    sigma2 = log_sigma2_32.exp()
    alpha = sigma2 / (mu32.pow(2) + _EPS)
    log_alpha = (alpha + _EPS).log()
    neg_kl = _K1 * torch.sigmoid(_K2 + _K3 * log_alpha) - 0.5 * torch.log1p(1.0 / (alpha + _EPS)) - _K1
    return -neg_kl  # positive KL (FP32)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _get_parent_and_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if isinstance(mod, nn.Linear) or cls in {"linear8bitlt", "linear4bit"}:
            yield name, mod

def _matches_any(name: str, tokens: Iterable[str]) -> bool:
    return any(t in name for t in tokens)

def _module_in_out(mod: nn.Module) -> Tuple[int, int]:
    if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
        return int(mod.out_features), int(mod.in_features)
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0]), int(w.shape[1])
    raise RuntimeError(f"Cannot infer in/out features for {mod.__class__.__name__}")

def _device_like(mod: nn.Module) -> torch.device:
    for p in mod.parameters(recurse=False):
        return p.device
    for b in mod.buffers(recurse=False):
        return b.device
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# Deterministic LoRA wrapper (no base copy)
# -----------------------------------------------------------------------------
class LoRAWrapper(nn.Module):
    def __init__(self, base: nn.Module, d_in: int, d_out: int, r: int, lora_alpha: float):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, int(r))

        dev = _device_like(base)
        self.A = nn.Parameter(torch.zeros(r, d_in, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.uniform_(self.B, a=-1e-3, b=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)
        delta = (x @ A.t()) @ B.t()
        return y + self.scaling * delta

# -----------------------------------------------------------------------------
# Variational LoRA wrapper (no base copy, ARD; QLoRA-safe)
# -----------------------------------------------------------------------------
class VariationalLoRAWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_in: int,
        d_out: int,
        r: int,
        lora_alpha: float = 1.0,
        tie_alpha_per_rank: bool = True,
        init_log_sigma2: float = -12.0,
        local_reparam: bool = False,
    ):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, int(r))
        self.tie_alpha_per_rank = bool(tie_alpha_per_rank)
        self.local_reparam = bool(local_reparam)
        self._sample = False  # toggled by set_sample

        dev = _device_like(base)

        # Posterior means — FP32 master params
        self.A_mu = nn.Parameter(torch.zeros(r, d_in, device=dev, dtype=torch.float32))
        self.B_mu = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A_mu, a=5 ** 0.5)
        nn.init.uniform_(self.B_mu, a=-1e-3, b=1e-3)

        # Posterior log-variances (FP32)
        if tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(torch.full((r,), init_log_sigma2, device=dev, dtype=torch.float32))
        else:
            self.A_log_sigma2 = nn.Parameter(torch.full((r, d_in), init_log_sigma2, device=dev, dtype=torch.float32))
            self.B_log_sigma2 = nn.Parameter(torch.full((d_out, r), init_log_sigma2, device=dev, dtype=torch.float32))

    # ---- utilities ----
    def set_sample(self, flag: bool):
        self._sample = bool(flag)

    def _log_sigma2_A(self):
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2.view(self.r, 1).expand_as(self.A_mu)
        return self.A_log_sigma2

    def _log_sigma2_B(self):
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2.view(1, self.r).expand_as(self.B_mu)
        return self.B_log_sigma2

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dtype, device = x.dtype, x.device

        A_mu32 = self.A_mu
        B_mu32 = self.B_mu
        A_log_s2_32 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)  # FP32
        B_log_s2_32 = torch.clamp(self._log_sigma2_B(), min=-20.0, max=2.0)  # FP32

        training_and_sample = self.training and self._sample
        if not training_and_sample:
            A_mu = A_mu32.to(dtype=dtype, device=device)
            B_mu = B_mu32.to(dtype=dtype, device=device)
            delta = (x @ A_mu.t()) @ B_mu.t()
            return y + self.scaling * delta

        if self.local_reparam:
            # ---- Molchanov (2017) local reparam trick ----
            x2 = x.pow(2).to(torch.float32)
            A_var = A_log_s2_32.exp()
            m_s = x.to(torch.float32) @ A_mu32.t()
            v_s = x2 @ A_var.t()

            B_var = B_log_s2_32.exp()
            m_y = m_s @ B_mu32.t()
            v_y = (v_s @ (B_mu32.pow(2)).t()) + ((m_s.pow(2)) @ B_var.t())

            eps = torch.randn_like(m_y, dtype=torch.float32, device=device)
            delta = (m_y + eps * torch.sqrt(v_y + 1e-8)).to(dtype=dtype)
        else:
            # Additive reparam
            scaleA = torch.exp(0.5 * A_log_s2_32)
            scaleB = torch.exp(0.5 * B_log_s2_32)
            epsA = torch.randn_like(A_mu32, dtype=torch.float32, device=device)
            epsB = torch.randn_like(B_mu32, dtype=torch.float32, device=device)
            A32 = A_mu32 + scaleA * epsA
            B32 = B_mu32 + scaleB * epsB
            A = A32.to(dtype=dtype, device=device)
            B = B32.to(dtype=dtype, device=device)
            delta = (x @ A.t()) @ B.t()

        return y + self.scaling * delta

    # ---- ARD bits ----
    def kl(self) -> torch.Tensor:
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        return _kl_from_mu_logs2(self.A_mu, A_log_s2).sum() + _kl_from_mu_logs2(self.B_mu, B_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        A_log_s2 = self._log_sigma2_A().float()
        B_log_s2 = self._log_sigma2_B().float()
        A_log_mu2 = (self.A_mu.float().pow(2) + _EPS).log()
        B_log_mu2 = (self.B_mu.float().pow(2) + _EPS).log()
        A_med = (A_log_s2 - A_log_mu2).median(dim=1).values
        B_med = (B_log_s2 - B_log_mu2).median(dim=0).values
        return 0.5 * (A_med + B_med)

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 1):
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        kmin = max(1, int(min_ranks))
        if keep.numel() < kmin:
            order = torch.argsort(la)
            keep = order[:kmin]
        self.A_mu = nn.Parameter(self.A_mu[keep])
        self.B_mu = nn.Parameter(self.B_mu[:, keep])
        if self.tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(self.rank_log_sigma2[keep])
        else:
            self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep])
            self.B_log_sigma2 = nn.Parameter(self.B_log_sigma2[:, keep])
        self.r = keep.numel()

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_bayeslora_to_llama(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    lora_alpha: float,
    kl_scale: float = 0.05,
    tie_alpha_per_rank: bool = True,
    local_reparam: bool = False,
    freeze_base: bool = True,
    exclude: Optional[List[str]] = None,
) -> nn.Module:
    exclude = exclude or []
    replaced = 0

    for name, lin in list(_iter_named_linears(model)):
        if not _matches_any(name, target_modules):
            continue
        if _matches_any(name, exclude):
            continue

        d_out, d_in = _module_in_out(lin)
        dev = _device_like(lin)

        # --- AFTER ---
        vl = VariationalLoRAWrapper(
            base=lin,
            d_in=d_in,
            d_out=d_out,
            r=int(rank),
            lora_alpha=float(lora_alpha),
            tie_alpha_per_rank=tie_alpha_per_rank,
            local_reparam=local_reparam,
        ).to(device=dev)  # keep base dtype (bf16/4bit), adapters are FP32 already

        vl._kl_scale = float(kl_scale)

        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, vl)
        replaced += 1

    if replaced == 0:
        print("[BayesLoRA] Warning: no target modules matched; check target_modules list.")
    else:
        print(f"[BayesLoRA] Wrapped {replaced} modules with VariationalLoRAWrapper (rank={rank}).")

    if freeze_base:
        for n, p in model.named_parameters():
            if not any(k in n for k in ("A_mu", "B_mu", "rank_log_sigma2", "A_log_sigma2", "B_log_sigma2")):
                p.requires_grad = False

    return model

@torch.no_grad()
def set_bayeslora_sampling(model: nn.Module, flag: bool):
    for m in model.modules():
        if hasattr(m, "set_sample"):
            m.set_sample(bool(flag))

def bayeslora_step(model: nn.Module, beta: float) -> torch.Tensor:
    kl_total = None
    kl_scale = 0.0
    device = None
    for m in model.modules():
        if hasattr(m, "kl") and callable(getattr(m, "kl")):
            kl = m.kl()
            device = kl.device
            kl_total = kl if kl_total is None else (kl_total + kl)
            mod_scale = getattr(m, "_kl_scale", None)
            if mod_scale is not None:
                kl_scale = max(kl_scale, float(mod_scale))
    if kl_total is None:
        if device is None:
            for p in model.parameters():
                device = p.device
                break
        return torch.tensor(0.0, device=device if device is not None else "cpu", dtype=torch.float32)
    if kl_scale == 0.0:
        kl_scale = 0.05
    return float(beta) * float(kl_scale) * kl_total

@torch.no_grad()
def bayeslora_prune(model: nn.Module, logalpha_thresh: float = 3.0, min_ranks: int = 1):
    for m in model.modules():
        if hasattr(m, "prune") and hasattr(m, "r"):
            m.prune(log_alpha_thresh=float(logalpha_thresh), min_ranks=int(min_ranks))

@torch.no_grad()
def get_adapter_ranks(model: nn.Module) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for name, m in model.named_modules():
        if isinstance(m, VariationalLoRAWrapper):
            ranks[name] = int(m.r)
    return ranks

@torch.no_grad()
def count_adapter_params(model: nn.Module) -> int:
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(k in n for k in ("A_mu", "B_mu", "rank_log_sigma2", "A_log_sigma2", "B_log_sigma2"))
    )
