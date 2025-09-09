import math
from dataclasses import dataclass
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Molchanov et al. 2017 constants
_K1, _K2, _K3 = 0.63576, 1.87320, 1.48695
_EPS = 1e-8

def kl_from_mu_logs2(mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    """Per-weight KL(q||p) using Molchanov et al. (2017) closed-form approx."""
    sigma2 = log_sigma2.exp()
    alpha = sigma2 / (mu.pow(2) + _EPS)
    log_alpha = (alpha + _EPS).log()
    neg_kl = _K1 * torch.sigmoid(_K2 + _K3 * log_alpha) - 0.5 * torch.log1p(1.0 / (alpha + _EPS)) - _K1
    return -neg_kl

def accuracy(model: nn.Module, loader, device, sample: bool = False) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, sample=sample) if "sample" in model.forward.__code__.co_varnames else model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total

def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=lr, weight_decay=weight_decay)

def default_prune_start_epoch(kl_freeze_epochs: int, beta_warmup_epochs: int) -> int:
    return kl_freeze_epochs + beta_warmup_epochs

@dataclass
class VLORASchedule:
    kl_freeze_epochs: int = 5
    beta_warmup_epochs: int = 10
    sample_warmup_epochs: int = 3

    def beta(self, epoch: int) -> float:
        if epoch <= self.kl_freeze_epochs:
            return 0.0
        t = epoch - self.kl_freeze_epochs
        return min(1.0, t / max(1, self.beta_warmup_epochs))

    def do_sample(self, epoch: int) -> bool:
        return epoch > self.sample_warmup_epochs