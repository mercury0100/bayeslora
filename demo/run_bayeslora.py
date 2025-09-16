#!/usr/bin/env python3
# lora_svd_vs_lora_mirror_fixed.py
# Vanilla LoRA vs Variational LoRA (Sparse Variational Dropout / ARD)
# - Correct MC sampling at eval
# - ELBO minibatch scaling per-sample (1/N)
# - LoRA scaling updated after rank pruning
# - Optional determinism & grad clipping
# - Log-α percentile diagnostics

import math
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ===== Molchanov et al. 2017 (Eq. 14) constants =====
_K1, _K2, _K3 = 0.63576, 1.87320, 1.48695
_EPS = 1e-8

def kl_from_mu_logs2(mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    """Per-weight KL(q||p) using the paper's closed-form approx with additive reparam."""
    sigma2 = log_sigma2.exp()
    alpha = sigma2 / (mu.pow(2) + _EPS)
    log_alpha = (alpha + _EPS).log()
    neg_kl = _K1 * torch.sigmoid(_K2 + _K3 * log_alpha) - 0.5 * torch.log1p(1.0 / (alpha + _EPS)) - _K1
    return -neg_kl  # positive KL, same shape as mu

# ---------- Minimal base Linear (avoid double-registration of base weights) ----------
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

# ---------- Vanilla LoRA ----------
class LoRALinear(nn.Module):
    """Deterministic LoRA: ΔW = B @ A (rank r)."""
    def __init__(self, d_in, d_out, r: int, lora_alpha: float = 1.0):
        super().__init__()
        self.base = _BaseLinear(d_in, d_out)
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / r
        self.A = nn.Parameter(torch.zeros(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.uniform_(self.B, a=-1e-3, b=1e-3)  # tiny but non-zero
    def forward(self, x):
        y = self.base(x)
        delta = (x @ self.A.t()) @ self.B.t()
        return y + self.scaling * delta

# ---------- Variational LoRA (SVD-ARD) ----------
class VariationalLoRALinear(nn.Module):
    """
    LoRA with Sparse Variational Dropout (ARD).
    Supports:
      - additive reparam (sampling A, B)
      - local reparam (sampling adapter output Δy) via --local-reparam
    """
    def __init__(self, d_in, d_out, r: int, lora_alpha: float = 1.0,
                 tie_alpha_per_rank: bool = True, init_log_sigma2: float = -12.0,
                 local_reparam: bool = False):
        super().__init__()
        self.base = _BaseLinear(d_in, d_out)
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / r
        self.tie_alpha_per_rank = tie_alpha_per_rank
        self.local_reparam = local_reparam

        # Posterior means
        self.A_mu = nn.Parameter(torch.zeros(r, d_in))
        self.B_mu = nn.Parameter(torch.zeros(d_out, r))

        # Posterior variances (log σ^2)
        if tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(torch.full((r,), init_log_sigma2))
        else:
            self.A_log_sigma2 = nn.Parameter(torch.full((r, d_in), init_log_sigma2))
            self.B_log_sigma2 = nn.Parameter(torch.full((d_out, r), init_log_sigma2))

        nn.init.kaiming_uniform_(self.A_mu, a=math.sqrt(5))
        nn.init.uniform_(self.B_mu, a=-1e-3, b=1e-3)

    def _log_sigma2_A(self):
        return self.rank_log_sigma2.view(self.r, 1).expand_as(self.A_mu) if self.tie_alpha_per_rank else self.A_log_sigma2
    def _log_sigma2_B(self):
        return self.rank_log_sigma2.view(1, self.r).expand_as(self.B_mu) if self.tie_alpha_per_rank else self.B_log_sigma2

    def forward(self, x, sample: bool = True):
        """
        If local_reparam:
            sample Δy in activation space with diag variance approximation.
        Else:
            sample A,B directly (additive reparam).
        Note: 'sample' controls stochasticity regardless of train/eval mode.
        """
        y = self.base(x)

        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        # Clamp variance logs a bit for numerical sanity
        A_log_s2 = torch.clamp(A_log_s2, min=-20.0, max=2.0)
        B_log_s2 = torch.clamp(B_log_s2, min=-20.0, max=2.0)

        if not sample:
            # Deterministic mean prediction
            delta = (x @ self.A_mu.t()) @ self.B_mu.t()
            return y + self.scaling * delta

        if self.local_reparam:
            # Local reparameterization (diag var approximation)
            x2 = x.pow(2)
            A_var = A_log_s2.exp()  # [r, d_in]
            m_s = x @ self.A_mu.t()                        # [B, r]
            v_s = x2 @ A_var.t()                           # [B, r]

            B_var = B_log_s2.exp()                         # [d_out, r]
            m_y = m_s @ self.B_mu.t()                      # [B, d_out]
            v_y = (v_s @ (self.B_mu.pow(2)).t()) + ((m_s.pow(2)) @ B_var.t())  # [B, d_out]
            eps = torch.randn_like(m_y)
            delta = m_y + eps * torch.sqrt(v_y + 1e-8)
        else:
            # Additive reparam: sample A and B directly
            A = self.A_mu + torch.exp(0.5 * A_log_s2) * torch.randn_like(self.A_mu)
            B = self.B_mu + torch.exp(0.5 * B_log_s2) * torch.randn_like(self.B_mu)
            delta = (x @ A.t()) @ B.t()

        return y + self.scaling * delta

    def kl(self) -> torch.Tensor:
        """Raw sum of per-weight KL (faithful to paper/ports)."""
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        return kl_from_mu_logs2(self.A_mu, A_log_s2).sum() + kl_from_mu_logs2(self.B_mu, B_log_s2).sum()

    @torch.no_grad()
    def rank_log_alpha(self) -> torch.Tensor:
        """Robust per-rank log α via median across dims for A[k,:], B[:,k]."""
        A_log_s2, B_log_s2 = self._log_sigma2_A(), self._log_sigma2_B()
        A_log_mu2 = (self.A_mu.pow(2) + _EPS).log()
        B_log_mu2 = (self.B_mu.pow(2) + _EPS).log()
        A_med = (A_log_s2 - A_log_mu2).median(dim=1).values  # [r]
        B_med = (B_log_s2 - B_log_mu2).median(dim=0).values  # [r]
        return 0.5 * (A_med + B_med)

    @torch.no_grad()
    def prune(self, log_alpha_thresh: float = 3.0):
        la = self.rank_log_alpha()
        keep = torch.nonzero(la < log_alpha_thresh, as_tuple=False).flatten()
        if keep.numel() == 0:
            keep = torch.tensor([torch.argmin(la)], device=la.device)
        self.A_mu = nn.Parameter(self.A_mu[keep])
        self.B_mu = nn.Parameter(self.B_mu[:, keep])
        if self.tie_alpha_per_rank:
            self.rank_log_sigma2 = nn.Parameter(self.rank_log_sigma2[keep])
        else:
            self.A_log_sigma2 = nn.Parameter(self.A_log_sigma2[keep])
            self.B_log_sigma2 = nn.Parameter(self.B_log_sigma2[:, keep])
        self.r = keep.numel()
        # Update LoRA scaling after rank change
        self.scaling = self.lora_alpha / self.r

    @torch.no_grad()
    def sparsity_report(self, log_alpha_thresh: float = 3.0):
        la = self.rank_log_alpha().detach().cpu()
        return {
            "ranks": int(self.r),
            "mean_log_alpha": float(la.mean()),
            "frac_ranks_over_thresh": float((la >= log_alpha_thresh).float().mean()),
        }

# ---------- Tiny 2-layer MLP with adapters ----------
class TinyMLP(nn.Module):
    def __init__(self, mode: str, d_in=2, d_hidden=128, num_classes=2,
                 rank=8, lora_alpha=8.0, tie_alpha_per_rank=True, freeze_base=False,
                 local_reparam=False):
        super().__init__()
        assert mode in {"vlora", "lora"}
        self.mode = mode
        if mode == "vlora":
            self.l1 = VariationalLoRALinear(d_in, d_hidden, r=rank, lora_alpha=lora_alpha,
                                            tie_alpha_per_rank=tie_alpha_per_rank,
                                            local_reparam=local_reparam)
            self.l2 = VariationalLoRALinear(d_hidden, num_classes, r=rank, lora_alpha=lora_alpha,
                                            tie_alpha_per_rank=tie_alpha_per_rank,
                                            local_reparam=local_reparam)
        else:
            self.l1 = LoRALinear(d_in, d_hidden, r=rank, lora_alpha=lora_alpha)
            self.l2 = LoRALinear(d_hidden, num_classes, r=rank, lora_alpha=lora_alpha)

        if freeze_base:
            for p in self.l1.base.parameters(): p.requires_grad = False
            for p in self.l2.base.parameters(): p.requires_grad = False

    def forward(self, x, sample=True):
        if self.mode == "vlora":
            x = F.relu(self.l1(x, sample=sample))
            x = self.l2(x, sample=sample)
        else:
            x = F.relu(self.l1(x))
            x = self.l2(x)
        return x

    def kl(self):
        if self.mode != "vlora":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.l1.kl() + self.l2.kl()

    def sparsity(self, thresh=3.0):
        if self.mode != "vlora": return None
        return {"layer1": self.l1.sparsity_report(thresh), "layer2": self.l2.sparsity_report(thresh)}

    @torch.no_grad()
    def prune(self, thresh=3.0):
        if self.mode != "vlora": return
        self.l1.prune(thresh); self.l2.prune(thresh)

# ---------- Data ----------
@dataclass
class TrainConfig:
    seed: int = 123
    n_train: int = 10000
    n_val: int = 2000
    batch_size: int = 256
    epochs: int = 25
    lr: float = 3e-3
    rank: int = 8
    lora_alpha: Optional[float] = None  # default None -> set to rank
    tie_alpha_per_rank: bool = True
    freeze_base: bool = False
    kl_freeze_epochs: int = 5
    beta_warmup_epochs: int = 10
    sample_warmup_epochs: int = 3
    kl_scale: float = 0.05         # acts like "reg_factor"
    logalpha_thresh: float = 3.0   # paper's default pruning cutoff
    local_reparam: bool = False
    deterministic: bool = False
    clip_grad: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def make_spiral(n, noise=0.2, turns=1.5):
    t = torch.linspace(0, 1, n)
    r = t
    th0 = turns * 2 * math.pi * t
    th1 = th0 + math.pi
    x0 = torch.stack([r * torch.cos(th0), r * torch.sin(th0)], 1)
    x1 = torch.stack([r * torch.cos(th1), r * torch.sin(th1)], 1)
    X = torch.cat([x0, x1], 0)
    y = torch.cat([torch.zeros(n), torch.ones(n)], 0).long()
    X += noise * torch.randn_like(X)
    return X, y

def make_loaders(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    Xtr, ytr = make_spiral(cfg.n_train // 2, noise=0.2)
    Xva, yva = make_spiral(cfg.n_val // 2, noise=0.2)
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)
    return tr, va

def accuracy(model, loader, device, sample=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, sample=sample)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total

# ---------- Train once ----------
def train_once(mode: str, cfg: TrainConfig, train_loader, val_loader, tag: str):
    device = torch.device(cfg.device)
    model = TinyMLP(
        mode=mode, d_in=2, d_hidden=128, num_classes=2,
        rank=cfg.rank, lora_alpha=(cfg.lora_alpha if cfg.lora_alpha is not None else cfg.rank),
        tie_alpha_per_rank=cfg.tie_alpha_per_rank, freeze_base=cfg.freeze_base,
        local_reparam=cfg.local_reparam
    ).to(device)

    # AdamW with NO weight decay on variational params (mirrors paper practice)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=0.0)

    B = len(train_loader)                # # of minibatches
    N = len(train_loader.dataset)        # train size (for per-sample scaling)

    print(f"\n=== Training {tag} ({mode.upper()}) on {device} ===")
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # β schedule with an initial KL freeze
        if mode == "vlora":
            if epoch <= cfg.kl_freeze_epochs:
                beta = 0.0
            else:
                t = epoch - cfg.kl_freeze_epochs
                beta = min(1.0, t / max(1, cfg.beta_warmup_epochs))
        else:
            beta = 0.0

        # Sampling starts after sample_warmup_epochs
        do_sample = (mode == "vlora") and (epoch > cfg.sample_warmup_epochs)

        tot_loss = tot_ce = tot_kl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, sample=do_sample)
            ce = F.cross_entropy(logits, yb, reduction="mean")
            if mode == "vlora":
                # Faithful ELBO per-sample scaling: loss = CE + β * reg * (KL_sum / N)
                kl_sum = model.kl()
                kl_term = cfg.kl_scale * (kl_sum / N)
                loss = ce + beta * kl_term
                tot_kl += float((kl_sum / N).item())
            else:
                loss = ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad is not None and cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad)
            opt.step()

            tot_loss += float(loss.item())
            tot_ce += float(ce.item())

        val_acc = accuracy(model, val_loader, device, sample=False)

        if mode == "vlora":
            sp = model.sparsity(thresh=cfg.logalpha_thresh)
            # Log-α percentiles for diagnostics
            with torch.no_grad():
                la1 = model.l1.rank_log_alpha().detach().cpu()
                la2 = model.l2.rank_log_alpha().detach().cpu()
                la = torch.cat([la1, la2], 0)
                p50, p90, p99 = torch.quantile(la, torch.tensor([0.5, 0.9, 0.99]))
            print(
                f"Epoch {epoch:02d} | loss {tot_loss/B:.4f} | ce {tot_ce/B:.4f} | "
                f"kl {tot_kl:.4f} | β {beta:.2f} | sample {int(do_sample)} | "
                f"val_acc {val_acc*100:.2f}% | "
                f"logα p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} | "
                f"L1 {sp['layer1']} | L2 {sp['layer2']}"
            )
        else:
            print(f"Epoch {epoch:02d} | loss {tot_loss/B:.4f} | ce {tot_ce/B:.4f} | val_acc {val_acc*100:.2f}%")

    # Eval & prune (VLORA) or just eval (LoRA)
    if mode == "vlora":
        acc_mean = accuracy(model, val_loader, device, sample=False)
        acc_mc   = accuracy(model, val_loader, device, sample=True)
        print(f"\n[{tag}] Validation accuracy (mean): {acc_mean*100:.2f}%")
        print(f"[{tag}] Validation accuracy (MC):   {acc_mc*100:.2f}%")

        print("\nPruning ranks with high log alpha…")
        model.prune(thresh=cfg.logalpha_thresh)
        sp = model.sparsity(thresh=cfg.logalpha_thresh)
        print(f"Post-prune ranks: L1={sp['layer1']['ranks']}  L2={sp['layer2']['ranks']}")
        acc_pruned = accuracy(model, val_loader, device, sample=False)
        print(f"[{tag}] Validation accuracy after pruning: {acc_pruned*100:.2f}%")
        return {"acc_mean": acc_mean, "acc_mc": acc_mc, "acc_pruned": acc_pruned}
    else:
        acc = accuracy(model, val_loader, device, sample=False)
        print(f"\n[{tag}] Validation accuracy (vanilla LoRA): {acc*100:.2f}%")
        return {"acc": acc}

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=None,
                   help="LoRA scaling; default = rank (so scaling=1)")
    p.add_argument("--freeze-base", action="store_true", default=False,
                   help="Freeze base Linear weights so adapters carry signal")
    p.add_argument("--no-tie-per-rank", action="store_true",
                   help="Disable per-rank alpha sharing (vlora only)")
    p.add_argument("--kl-freeze-epochs", type=int, default=5,
                   help="Epochs with KL fully off (β=0) before warm-up")
    p.add_argument("--beta-warmup-epochs", type=int, default=10)
    p.add_argument("--sample-warmup-epochs", type=int, default=3)
    p.add_argument("--kl-scale", type=float, default=0.05,
                   help="KL multiplier (aka reg_factor). Loss = CE + β * kl_scale * (sum(KL)/N)")
    p.add_argument("--logalpha-thresh", type=float, default=3.0,
                   help="Paper's pruning cutoff (log alpha > 3 → prune)")
    p.add_argument("--local-reparam", action="store_true", default=False,
                   help="Use local reparameterization (sample in activation space)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="Enable deterministic algorithms (slower).")
    p.add_argument("--clip-grad", type=float, default=None,
                   help="If set, clip total grad norm to this value.")
    args = p.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank,
        lora_alpha=(args.lora_alpha if args.lora_alpha is not None else args.rank),
        tie_alpha_per_rank=not args.no_tie_per_rank,
        freeze_base=args.freeze_base,
        kl_freeze_epochs=args.kl_freeze_epochs,
        beta_warmup_epochs=args.beta_warmup_epochs,
        sample_warmup_epochs=args.sample_warmup_epochs,
        kl_scale=args.kl_scale,
        logalpha_thresh=args.logalpha_thresh,
        local_reparam=args.local_reparam,
        seed=args.seed,
        deterministic=args.deterministic,
        clip_grad=args.clip_grad,
    )

    # Seeding / determinism
    torch.manual_seed(cfg.seed)
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    train_loader, val_loader = make_loaders(cfg)

    # Train Variational LoRA (SVD-ARD)
    res_v = train_once("vlora", cfg, train_loader, val_loader, tag="Variational LoRA (SVD-ARD)")

    # Re-seed for comparable init
    torch.manual_seed(cfg.seed)

    # Train vanilla LoRA baseline
    res_b = train_once("lora", cfg, train_loader, val_loader, tag="Vanilla LoRA Baseline")

    print("\n=== Summary ===")
    print(f"Variational LoRA (mean):   {res_v['acc_mean']*100:.2f}%")
    print(f"Variational LoRA (MC):     {res_v['acc_mc']*100:.2f}%")
    print(f"Variational LoRA (pruned): {res_v['acc_pruned']*100:.2f}%")
    print(f"Vanilla LoRA baseline:     {res_b['acc']*100:.2f}%")

if __name__ == "__main__":
    main()