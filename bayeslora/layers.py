import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_from_mu_logs2

_EPS = 1e-8

# -------- Base wrappers that keep a frozen (or trainable) base op and add a LoRA delta --------

class _BaseLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _BaseConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


# -------------------------- Deterministic LoRA --------------------------

class LoRALinear(nn.Module):
    """ΔW = B @ A (rank r)."""
    def __init__(self, d_in, d_out, r: int, lora_alpha: float = 1.0, freeze_base: bool = True):
        super().__init__()
        self.base = _BaseLinear(d_in, d_out)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.r)

        self.A = nn.Parameter(torch.zeros(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.uniform_(self.B, a=-1e-3, b=1e-3)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        delta = (x @ self.A.t()) @ self.B.t()
        return y + self.scaling * delta


class LoRAConv2d(nn.Module):
    """
    Channel-low-rank LoRA for Conv2d via (1x1 down) + (kxk up).
    Keeps the spatial kernel in the 'up' conv, ranks across channels.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, r: int = 8, lora_alpha: float = 1.0, freeze_base: bool = True):
        super().__init__()
        self.base = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.r)

        # Down: in_ch -> r (1x1), Up: r -> out_ch (kxk)
        self.A = nn.Conv2d(in_ch, self.r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(self.r, out_ch, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.uniform_(self.B.weight, a=-1e-3, b=1e-3)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        delta = self.B(self.A(x))
        return y + self.scaling * delta


# -------------------------- Variational LoRA (BayesLoRA) --------------------------

class _VLORA_Mixin:
    """Shared ARD bits for Linear/Conv variants."""

    def kl(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 1):
        raise NotImplementedError

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sparsity_report(self, log_alpha_thresh: float = 3.0):
        la = self.rank_log_alpha().detach().cpu()
        return {
            "ranks": int(self.r),
            "mean_log_alpha": float(la.mean()),
            "frac_ranks_over_thresh": float((la >= log_alpha_thresh).float().mean()),
        }


class VariationalLoRALinear(nn.Module, _VLORA_Mixin):
    """
    BayesLoRA (variational LoRA with ARD) for Linear layers.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        r: int,
        lora_alpha: float = 1.0,
        tie_alpha_per_rank: bool = True,
        init_log_sigma2: float = -12.0,
        local_reparam: bool = False,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = _BaseLinear(d_in, d_out)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.r)
        self.tie_alpha_per_rank = bool(tie_alpha_per_rank)
        self.local_reparam = bool(local_reparam)

        # posterior means
        self.A_mu = nn.Parameter(torch.zeros(self.r, d_in))
        self.B_mu = nn.Parameter(torch.zeros(d_out, self.r))

        # posterior log variances
        if self.tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(torch.full((self.r,), init_log_sigma2))
        else:
            self.A_log_sigma2 = nn.Parameter(torch.full((self.r, d_in), init_log_sigma2))
            self.B_log_sigma2 = nn.Parameter(torch.full((d_out, self.r), init_log_sigma2))

        nn.init.kaiming_uniform_(self.A_mu, a=math.sqrt(5))
        nn.init.uniform_(self.B_mu, a=-1e-3, b=1e-3)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self._sample_default = True

    def set_sample(self, flag: bool):
        self._sample_default = bool(flag)

    def _log_sigma2_A(self) -> torch.Tensor:
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2.view(self.r, 1).expand_as(self.A_mu)
        return self.A_log_sigma2

    def _log_sigma2_B(self) -> torch.Tensor:
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2.view(1, self.r).expand_as(self.B_mu)
        return self.B_log_sigma2

    def forward(self, x: torch.Tensor, sample: Optional[bool] = None) -> torch.Tensor:
        if sample is None:
            sample = self._sample_default

        y = self.base(x)
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        A_log_s2 = torch.clamp(A_log_s2, min=-20.0, max=2.0)
        B_log_s2 = torch.clamp(B_log_s2, min=-20.0, max=2.0)

        if not sample:
            delta = (x @ self.A_mu.t()) @ self.B_mu.t()
            return y + self.scaling * delta

        if self.local_reparam:
            # simplified local reparam (underestimates variance slightly)
            x2 = x.pow(2)
            A_var = A_log_s2.exp()
            m_s = x @ self.A_mu.t()
            v_s = x2 @ A_var.t()
            B_var = B_log_s2.exp()
            m_y = m_s @ self.B_mu.t()
            v_y = (v_s @ (self.B_mu.pow(2)).t()) + ((m_s.pow(2)) @ B_var.t())
            eps = torch.randn_like(m_y)
            delta = m_y + eps * torch.sqrt(v_y + 1e-8)
        else:
            A = self.A_mu + torch.exp(0.5 * A_log_s2) * torch.randn_like(self.A_mu)
            B = self.B_mu + torch.exp(0.5 * B_log_s2) * torch.randn_like(self.B_mu)
            delta = (x @ A.t()) @ B.t()

        return y + self.scaling * delta

    def kl(self) -> torch.Tensor:
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        return kl_from_mu_logs2(self.A_mu, A_log_s2).sum() + kl_from_mu_logs2(self.B_mu, B_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        A_log_mu2 = (self.A_mu.pow(2) + _EPS).log()
        B_log_mu2 = (self.B_mu.pow(2) + _EPS).log()
        A_med = (A_log_s2 - A_log_mu2).median(dim=1).values
        B_med = (B_log_s2 - B_log_mu2).median(dim=0).values
        return 0.5 * (A_med + B_med)

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 1):
        device = self.A_mu.device
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        if keep.numel() < min_ranks:
            _, idx = torch.topk(-la, k=min_ranks)
            keep = idx
        keep = keep.sort().values

        self.A_mu = nn.Parameter(self.A_mu[keep].to(device))
        self.B_mu = nn.Parameter(self.B_mu[:, keep].to(device))
        if self.tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(self.rank_log_sigma2[keep].to(device))
        else:
            self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep].to(device))
            self.B_log_sigma2 = nn.Parameter(self.B_log_sigma2[:, keep].to(device))

        self.r = int(keep.numel())
        self.scaling = self.lora_alpha / max(1, self.r)


class VariationalLoRAConv2d(nn.Module, _VLORA_Mixin):
    """
    BayesLoRA for Conv2d using (1x1 down, kxk up) adapters with ARD on both convs.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, r: int = 8, lora_alpha: float = 1.0, tie_alpha_per_rank: bool = True,
                 init_log_sigma2: float = -12.0, local_reparam: bool = False, freeze_base: bool = True):
        super().__init__()
        self.base = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.r)
        self.tie_alpha_per_rank = tie_alpha_per_rank
        self.local_reparam = local_reparam

        self.A_mu = nn.Conv2d(in_ch, self.r, kernel_size=1, bias=False)
        self.B_mu = nn.Conv2d(self.r, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)

        if tie_alpha_per_rank:
            self.rank_log_sigma2_A = nn.Parameter(torch.full((self.r,), init_log_sigma2))
            self.rank_log_sigma2_B = nn.Parameter(torch.full((self.r,), init_log_sigma2))
        else:
            self.A_log_sigma2 = nn.Parameter(torch.full((self.r, in_ch, 1, 1), init_log_sigma2))
            self.B_log_sigma2 = nn.Parameter(torch.full((out_ch, self.r, kernel_size, kernel_size), init_log_sigma2))

        nn.init.kaiming_uniform_(self.A_mu.weight, a=math.sqrt(5))
        nn.init.uniform_(self.B_mu.weight, a=-1e-3, b=1e-3)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self._sample_default = True

    def set_sample(self, flag: bool):
        self._sample_default = bool(flag)

    def _log_sigma2_A(self):
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2_A.view(self.r, 1, 1, 1).expand_as(self.A_mu.weight)
        return self.A_log_sigma2

    def _log_sigma2_B(self):
        if self.tie_alpha_per_rank:
            return self.rank_log_sigma2_B.view(1, self.r, 1, 1).expand_as(self.B_mu.weight)
        return self.B_log_sigma2

    def forward(self, x, sample: Optional[bool] = None):
        if sample is None:
            sample = self._sample_default

        y = self.base(x)
        A_log_s2 = torch.clamp(self._log_sigma2_A(), min=-20.0, max=2.0)
        B_log_s2 = torch.clamp(self._log_sigma2_B(), min=-20.0, max=2.0)

        if not sample:
            delta = self.B_mu(self.A_mu(x))
            return y + self.scaling * delta

        # Additive reparam for convs
        A = self.A_mu.weight + torch.exp(0.5 * A_log_s2) * torch.randn_like(self.A_mu.weight)
        B = self.B_mu.weight + torch.exp(0.5 * B_log_s2) * torch.randn_like(self.B_mu.weight)

        s = F.conv2d(x, A, bias=None, stride=1, padding=0, dilation=1, groups=1)
        delta = F.conv2d(s, B, bias=None, stride=self.B_mu.stride, padding=self.B_mu.padding,
                         dilation=self.B_mu.dilation, groups=self.B_mu.groups)
        return y + self.scaling * delta

    def kl(self) -> torch.Tensor:
        A_log_s2 = self._log_sigma2_A()
        B_log_s2 = self._log_sigma2_B()
        return kl_from_mu_logs2(self.A_mu.weight, A_log_s2).sum() + kl_from_mu_logs2(self.B_mu.weight, B_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        A_log_s2 = self._log_sigma2_A()
        B_log_s2 = self._log_sigma2_B()
        A_log_mu2 = (self.A_mu.weight.pow(2) + _EPS).log()
        B_log_mu2 = (self.B_mu.weight.pow(2) + _EPS).log()
        A_med = (A_log_s2 - A_log_mu2).flatten(1).median(dim=1).values
        B_med = (B_log_s2 - B_log_mu2).permute(1, 0, 2, 3).contiguous().flatten(1).median(dim=1).values
        return 0.5 * (A_med + B_med)

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0, min_ranks: int = 1):
        device = self.A_mu.weight.device
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        if keep.numel() < min_ranks:
            _, idx = torch.topk(-la, k=min_ranks)
            keep = idx
        keep = keep.sort().values

        old_A_w = self.A_mu.weight.detach()
        old_B_w = self.B_mu.weight.detach()
        new_r = int(keep.numel())

        new_A = nn.Conv2d(self.A_mu.in_channels, new_r, kernel_size=1, bias=False).to(device)
        new_B = nn.Conv2d(new_r, self.B_mu.out_channels, kernel_size=self.B_mu.kernel_size,
                          stride=self.B_mu.stride, padding=self.B_mu.padding,
                          dilation=self.B_mu.dilation, groups=self.B_mu.groups, bias=False).to(device)

        new_A.weight.copy_(old_A_w[keep, :, :, :].to(device))
        new_B.weight.copy_(old_B_w[:, keep, :, :].to(device))

        self.A_mu = new_A
        self.B_mu = new_B

        if self.tie_alpha_per_rank:
            self.rank_log_sigma2_A = nn.Parameter(self.rank_log_sigma2_A[keep].to(device))
            self.rank_log_sigma2_B = nn.Parameter(self.rank_log_sigma2_B[keep].to(device))
        else:
            self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep].to(device))
            self.B_log_sigma2 = nn.Parameter(self.B_log_sigma2[:, keep].to(device))

        self.r = new_r
        self.scaling = self.lora_alpha / self.r
