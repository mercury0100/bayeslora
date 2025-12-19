from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

_K1, _K2, _K3 = 0.63576, 1.87320, 1.48695
_EPS = 1e-8


def _kl_from_mu_logs2(mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    mu32 = mu.float()
    log_sigma2_32 = log_sigma2.float()
    sigma2 = log_sigma2_32.exp()
    alpha = sigma2 / (mu32.pow(2) + _EPS)
    log_alpha = (alpha + _EPS).log()
    neg_kl = _K1 * torch.sigmoid(_K2 + _K3 * log_alpha) - 0.5 * torch.log1p(1.0 / (alpha + _EPS)) - _K1
    return -neg_kl


def _get_parent_and_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _iter_named_linears(model: nn.Module):
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if isinstance(mod, nn.Linear) or cls in {"linear8bitlt", "linear4bit"} or cls == "conv1d":
            yield name, mod


def _module_in_out(mod: nn.Module) -> Tuple[int, int]:
    cls = mod.__class__.__name__.lower()
    if cls == "conv1d" and "transformers" in getattr(mod, "__module__", ""):
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            in_f, out_f = int(w.shape[0]), int(w.shape[1])
            return out_f, in_f
    if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
        return int(mod.out_features), int(mod.in_features)
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0]), int(w.shape[1])
    raise RuntimeError(f"Cannot infer in/out features for {mod.__class__.__name__}")


def _matches_any(name: str, tokens: Iterable[str]) -> bool:
    return any(t in name for t in tokens)


def _device_like(mod: nn.Module) -> torch.device:
    for p in mod.parameters(recurse=False):
        return p.device
    for b in mod.buffers(recurse=False):
        return b.device
    return torch.device("cpu")


class LoRAWrapper(nn.Module):
    def __init__(self, base: nn.Module, d_in: int, d_out: int, r: int, lora_alpha: float):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, int(r))
        dev = _device_like(base)
        self.A = nn.Parameter(torch.zeros(r, d_in, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        if r > 0:
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            nn.init.uniform_(self.B, a=-1e-3, b=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A = self.A.to(dtype=x.dtype, device=x.device)
        B = self.B.to(dtype=x.dtype, device=x.device)
        delta = (x @ A.t()) @ B.t()
        return y + self.scaling * delta


class VariationalLoRAWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_in: int,
        d_out: int,
        r: int,
        lora_alpha: float = 1.0,
        tie_alpha_per_rank: bool = True,
        init_log_sigma2: float = -8.0,
        local_reparam: bool = False,
    ):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, int(r))
        self.tie_alpha_per_rank = bool(tie_alpha_per_rank)
        self.local_reparam = bool(local_reparam)
        self._sample = False

        dev = _device_like(base)
        self.A_mu = nn.Parameter(torch.zeros(r, d_in, device=dev, dtype=torch.float32))
        self.B_mu = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        if r > 0:
            nn.init.kaiming_uniform_(self.A_mu, a=5 ** 0.5)
            nn.init.uniform_(self.B_mu, a=-1e-3, b=1e-3)

        if tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(torch.full((r,), init_log_sigma2, device=dev, dtype=torch.float32))
        else:
            self.A_log_sigma2 = nn.Parameter(torch.full((r, d_in), init_log_sigma2, device=dev, dtype=torch.float32))
            self.B_log_sigma2 = nn.Parameter(torch.full((d_out, r), init_log_sigma2, device=dev, dtype=torch.float32))

        self._kl_scale = 0.05

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dtype, device = x.dtype, x.device
        A_mu32 = self.A_mu
        B_mu32 = self.B_mu
        A_log_s2_32 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)
        B_log_s2_32 = torch.clamp(self._log_sigma2_B(), min=-20.0, max=2.0)
        if not self._sample:
            A_mu = A_mu32.to(dtype=dtype, device=device)
            B_mu = B_mu32.to(dtype=dtype, device=device)
            delta = (x @ A_mu.t()) @ B_mu.t()
            return y + self.scaling * delta
        if self.local_reparam:
            x2 = x.pow(2).to(torch.float32)
            A_var = A_log_s2_32.exp()
            m_s = x.to(torch.float32) @ A_mu32.t()
            v_s = x2 @ A_var.t()
            B_var = B_log_s2_32.exp()
            m_y = m_s @ B_mu32.t()
            v_y = (v_s @ (B_mu32.pow(2)).t()) + ((m_s.pow(2)) @ B_var.t()) + (v_s @ B_var.t())
            v_y = torch.clamp(v_y, min=0.0)
            eps = torch.randn_like(m_y, dtype=torch.float32, device=device)
            delta = (m_y + eps * torch.sqrt(v_y + 1e-8)).to(dtype=dtype)
        else:
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

    @torch.no_grad()
    def fix_gauge(self, eps: float = 1e-12) -> None:
        if int(self.r) == 0:
            return
        A = self.A_mu
        B = self.B_mu
        nA = torch.linalg.vector_norm(A, ord=2, dim=1)
        nB = torch.linalg.vector_norm(B, ord=2, dim=0)
        c = torch.sqrt((nB + eps) / (nA + eps))
        A.mul_(c.unsqueeze(1))
        B.mul_(1.0 / c.unsqueeze(0))
        if not self.tie_alpha_per_rank:
            logc = torch.log(c + eps)
            self.A_log_sigma2.add_(2.0 * logc.unsqueeze(1))
            self.B_log_sigma2.add_(-2.0 * logc.unsqueeze(0))

    def kl(self) -> torch.Tensor:
        if int(self.r) == 0:
            return torch.tensor(0.0, device=self.A_mu.device, dtype=torch.float32)
        A_log_s2 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)
        B_log_s2 = torch.clamp(self._log_sigma2_B(), min=-20.0, max=2.0)
        return _kl_from_mu_logs2(self.A_mu, A_log_s2).sum() + _kl_from_mu_logs2(self.B_mu, B_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        if int(self.r) == 0:
            return torch.empty(0, device=self.A_mu.device, dtype=torch.float32)
        A_log_s2 = self._log_sigma2_A().float()
        B_log_s2 = self._log_sigma2_B().float()
        A_log_mu2 = (self.A_mu.float().pow(2) + _EPS).log()
        B_log_mu2 = (self.B_mu.float().pow(2) + _EPS).log()
        A_med = (A_log_s2 - A_log_mu2).median(dim=1).values
        B_med = (B_log_s2 - B_log_mu2).median(dim=0).values
        return 0.5 * (A_med + B_med)

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 0) -> bool:
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        kmin = max(0, int(min_ranks))
        if keep.numel() < kmin:
            order = torch.argsort(la)
            keep = order[:kmin]
        changed = keep.numel() != int(self.r)
        if not changed:
            return False
        # Replace parameters (optimizer must be recreated by caller)
        if keep.numel() == 0:
            dev = self.A_mu.device
            dt = self.A_mu.dtype
            din = self.A_mu.shape[1]
            dout = self.B_mu.shape[0]
            self.A_mu = nn.Parameter(torch.zeros(0, din, device=dev, dtype=dt))
            self.B_mu = nn.Parameter(torch.zeros(dout, 0, device=dev, dtype=dt))
            if self.tie_alpha_per_rank:
                self.rank_log_sigma2 = nn.Parameter(torch.zeros(0, device=dev, dtype=dt))
            else:
                self.A_log_sigma2 = nn.Parameter(torch.zeros(0, din, device=dev, dtype=dt))
                self.B_log_sigma2 = nn.Parameter(torch.zeros(dout, 0, device=dev, dtype=dt))
        else:
            self.A_mu = nn.Parameter(self.A_mu[keep])
            self.B_mu = nn.Parameter(self.B_mu[:, keep])
            if self.tie_alpha_per_rank:
                self.rank_log_sigma2 = nn.Parameter(self.rank_log_sigma2[keep])
            else:
                self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep])
                self.B_log_sigma2 = nn.Parameter(self.B_log_sigma2[:, keep])
        self.r = keep.numel()
        return True

class AsymVariationalLoRAWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_in: int,
        d_out: int,
        r: int,
        lora_alpha: float = 1.0,
        tie_alpha_per_rank: bool = True,
        init_log_sigma2: float = -8.0,
        local_reparam: bool = False,
    ):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, int(r))
        self.tie_alpha_per_rank = bool(tie_alpha_per_rank)
        self.local_reparam = bool(local_reparam)
        self._sample = False

        dev = _device_like(base)
        # Means
        self.A_mu = nn.Parameter(torch.zeros(r, d_in, device=dev, dtype=torch.float32))
        self.B_mu = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        if r > 0:
            nn.init.kaiming_uniform_(self.A_mu, a=5 ** 0.5)
            nn.init.uniform_(self.B_mu, a=-1e-3, b=1e-3)

        # --- Asymmetric Bayes: variance only for A ---
        if tie_alpha_per_rank:
            # One log-variance per rank, broadcast over A's row
            self.rank_log_sigma2 = nn.Parameter(
                torch.full((r,), init_log_sigma2, device=dev, dtype=torch.float32)
            )
        else:
            self.A_log_sigma2 = nn.Parameter(
                torch.full((r, d_in), init_log_sigma2, device=dev, dtype=torch.float32)
            )

        self._kl_scale = 0.05

    def set_sample(self, flag: bool):
        self._sample = bool(flag)

    def _log_sigma2_A(self):
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2.view(self.r, 1).expand_as(self.A_mu)
        return self.A_log_sigma2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dtype, device = x.dtype, x.device
        A_mu32 = self.A_mu
        B_mu32 = self.B_mu

        # Clamp A variances
        A_log_s2_32 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)

        if not self._sample:
            # Deterministic mean prediction
            A_mu = A_mu32.to(dtype=dtype, device=device)
            B_mu = B_mu32.to(dtype=dtype, device=device)
            delta = (x @ A_mu.t()) @ B_mu.t()
            return y + self.scaling * delta

        # --- Sampling path ---
        if self.local_reparam:
            # Local reparameterisation with stochastic A and deterministic B
            x_f = x.to(torch.float32)
            x2 = x_f.pow(2)
            A_var = A_log_s2_32.exp()              # [r, d_in]

            # Rank activations
            m_s = x_f @ A_mu32.t()                 # [B, r]
            v_s = x2 @ A_var.t()                   # [B, r]

            # Project through deterministic B
            m_y = m_s @ B_mu32.t()                 # [B, d_out]
            v_y = v_s @ (B_mu32.pow(2)).t()        # [B, d_out]
            v_y = torch.clamp(v_y, min=0.0)

            eps = torch.randn_like(m_y, dtype=torch.float32, device=device)
            delta = (m_y + eps * torch.sqrt(v_y + 1e-8)).to(dtype=dtype)
        else:
            # Direct weight sampling: A stochastic, B deterministic
            scaleA = torch.exp(0.5 * A_log_s2_32)
            epsA = torch.randn_like(A_mu32, dtype=torch.float32, device=device)
            A32 = A_mu32 + scaleA * epsA
            A = A32.to(dtype=dtype, device=device)
            B = B_mu32.to(dtype=dtype, device=device)
            delta = (x @ A.t()) @ B.t()

        return y + self.scaling * delta

    @torch.no_grad()
    def fix_gauge(self, eps: float = 1e-12) -> None:
        if int(self.r) == 0:
            return
        A = self.A_mu
        B = self.B_mu
        nA = torch.linalg.vector_norm(A, ord=2, dim=1)
        nB = torch.linalg.vector_norm(B, ord=2, dim=0)
        c = torch.sqrt((nB + eps) / (nA + eps))
        A.mul_(c.unsqueeze(1))
        B.mul_(1.0 / c.unsqueeze(0))
        if not self.tie_alpha_per_rank:
            # Only A has variances now
            logc = torch.log(c + eps)
            self.A_log_sigma2.add_(2.0 * logc.unsqueeze(1))

    def kl(self) -> torch.Tensor:
        """KL(q(A) || p(A)) with Molchanov prior, B deterministic."""
        if int(self.r) == 0:
            return torch.tensor(0.0, device=self.A_mu.device, dtype=torch.float32)
        A_log_s2 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)
        return _kl_from_mu_logs2(self.A_mu, A_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        """Rank-wise log alpha from A only."""
        if int(self.r) == 0:
            return torch.empty(0, device=self.A_mu.device, dtype=torch.float32)
        A_log_s2 = self._log_sigma2_A().float()
        A_log_mu2 = (self.A_mu.float().pow(2) + _EPS).log()
        # Median over input dims → per-rank log alpha
        A_med = (A_log_s2 - A_log_mu2).median(dim=1).values
        return A_med

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 0) -> bool:
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        kmin = max(0, int(min_ranks))
        if keep.numel() < kmin:
            order = torch.argsort(la)
            keep = order[:kmin]

        changed = keep.numel() != int(self.r)
        if not changed:
            return False

        # Replace parameters (optimizer must be recreated by caller)
        dev = self.A_mu.device
        dt = self.A_mu.dtype
        din = self.A_mu.shape[1]
        dout = self.B_mu.shape[0]

        if keep.numel() == 0:
            self.A_mu = nn.Parameter(torch.zeros(0, din, device=dev, dtype=dt))
            self.B_mu = nn.Parameter(torch.zeros(dout, 0, device=dev, dtype=dt))
            if self.tie_alpha_per_rank:
                self.rank_log_sigma2 = nn.Parameter(torch.zeros(0, device=dev, dtype=dt))
            else:
                self.A_log_sigma2 = nn.Parameter(torch.zeros(0, din, device=dev, dtype=dt))
        else:
            self.A_mu = nn.Parameter(self.A_mu[keep])
            self.B_mu = nn.Parameter(self.B_mu[:, keep])
            if self.tie_alpha_per_rank:
                self.rank_log_sigma2 = nn.Parameter(self.rank_log_sigma2[keep])
            else:
                self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep])
        self.r = keep.numel()
        return True


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_bayeslora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    lora_alpha: float,
    kl_scale: float = 1e-6,
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
    """Return beta * sum_i (kl_scale_i * KL_i). If no adapters, return 0 tensor."""
    total = None
    device = None
    for m in model.modules():
        if hasattr(m, "kl") and callable(getattr(m, "kl")):
            kl = m.kl()
            device = kl.device
            scale = float(getattr(m, "_kl_scale", 0.05))
            term = scale * kl
            total = term if total is None else (total + term)
    if total is None:
        if device is None:
            for p in model.parameters():
                device = p.device
                break
        return torch.tensor(0.0, device=device if device is not None else "cpu", dtype=torch.float32)
    return float(beta) * total


@torch.no_grad()
def bayeslora_prune(model: nn.Module, logalpha_thresh: float = 3.0, min_ranks: int = 0) -> bool:
    """Prune all adapter modules. Returns True if any module changed its rank.
    IMPORTANT: If True, you must rebuild the optimizer to sync parameter groups.
    """
    changed_any = False
    for m in model.modules():
        if hasattr(m, "prune") and hasattr(m, "r"):
            changed_any = m.prune(log_alpha_thresh=float(logalpha_thresh), min_ranks=int(min_ranks)) or changed_any
    return changed_any


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
