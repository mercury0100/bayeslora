#!/usr/bin/env python3
"""
Benchmark Vanilla LoRA vs BayesLoRA on CIFAR-10 with *pretrained* ResNet-18.

Example:
  python benchmark_resnet_cifar.py \
    python benchmark_resnet_cifar.py --epochs 20 --rank 128 --prune-every 3 --min-ranks 2
"""
import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# package imports
from bayeslora import apply_lora_adapters, accuracy, make_optimizer, default_prune_start_epoch

# --------- Robust pretrained weights handling across torchvision versions ---------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_resnet18_pretrained():
    """Return a pretrained ResNet-18 regardless of torchvision version."""
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = getattr(ResNet18_Weights, "IMAGENET1K_V1", ResNet18_Weights.DEFAULT)
        return resnet18(weights=weights)
    except Exception:
        # Older API: weights="IMAGENET1K_V1" or pretrained=True
        try:
            from torchvision.models import resnet18
            return resnet18(weights="IMAGENET1K_V1")
        except Exception:
            return resnet18(pretrained=True)

# ----------------------------------------------------------------------------------

@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4

    rank: int = 8
    lora_alpha: Optional[float] = None
    freeze_base: bool = True
    tie_alpha_per_rank: bool = True
    local_reparam: bool = False

    # BayesLoRA schedules
    kl_scale: float = 0.05
    kl_freeze_epochs: int = 5
    beta_warmup_epochs: int = 10
    sample_warmup_epochs: int = 3
    logalpha_thresh: float = 3.0

    # prune-as-you-go
    prune_every: int = 0
    prune_start_epoch: Optional[int] = None
    min_ranks: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123


def make_loaders(batch_size: int):
    # Use fixed ImageNet normalization (works for all torchvision versions)
    tf_train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)
    tr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return tr, te


def set_bayeslora_sampling(model: nn.Module, flag: bool):
    """Toggle stochastic sampling for BayesLoRA modules that expose `set_sample`."""
    for m in model.modules():
        if hasattr(m, "set_sample"):
            m.set_sample(flag)


def train_epoch(model, loader, opt, device, cfg: Config, epoch: int, variant: str, N: int):
    model.train()
    # sampling policy per epoch
    do_sample = (variant == "vlora") and (epoch > cfg.sample_warmup_epochs)
    set_bayeslora_sampling(model, do_sample)

    # β schedule for BayesLoRA
    if variant == "vlora":
        if epoch <= cfg.kl_freeze_epochs:
            beta = 0.0
        else:
            t = epoch - cfg.kl_freeze_epochs
            beta = min(1.0, t / max(1, cfg.beta_warmup_epochs))
    else:
        beta = 0.0

    tot_loss = tot_ce = tot_kl = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)  # sampling controlled at module level
        ce = F.cross_entropy(logits, yb, reduction="mean")

        if variant == "vlora":
            kl_sum = 0.0
            for m in model.modules():
                if hasattr(m, "kl") and callable(getattr(m, "kl")):
                    try:
                        kl_sum = kl_sum + m.kl()
                    except Exception:
                        pass
            kl_term = cfg.kl_scale * (kl_sum / N)  # per-sample scaling
            loss = ce + beta * kl_term
            tot_kl += float((kl_sum / N).item())
        else:
            loss = ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tot_loss += float(loss.item())
        tot_ce += float(ce.item())

    return tot_loss / len(loader), tot_ce / len(loader), tot_kl, beta, do_sample


def prune_as_you_go(model, cfg: Config, epoch: int):
    if cfg.prune_every <= 0:
        return False, (0, 0)
    start = cfg.prune_start_epoch if cfg.prune_start_epoch is not None else default_prune_start_epoch(
        cfg.kl_freeze_epochs, cfg.beta_warmup_epochs
    )
    if epoch < start or (epoch % cfg.prune_every) != 0:
        return False, (0, 0)

    before = after = 0
    for m in model.modules():
        if hasattr(m, "prune"):
            if hasattr(m, "r"):
                before += int(m.r)
            try:
                m.prune(log_alpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
            except TypeError:
                pass  # deterministic LoRA layers won't have ARD prune
            if hasattr(m, "r"):
                after += int(m.r)
    return True, (before, after)


def build_pretrained_resnet18_with_adapters(variant: str, cfg: Config):
    """
    Load pretrained ResNet-18 (ImageNet), swap classifier to 10 classes, then wrap
    with LoRA / BayesLoRA adapters (Conv+Linear).
    We keep the original 7x7/stride-2 stem + maxpool to match pretrained filters.
    """
    model = load_resnet18_pretrained()
    model.fc = nn.Linear(model.fc.in_features, 10)  # replace classifier for CIFAR-10

    # Apply adapters
    model = apply_lora_adapters(
        model,
        variant=variant,
        rank=cfg.rank,
        lora_alpha=(cfg.lora_alpha if cfg.lora_alpha is not None else cfg.rank),
        freeze_base=cfg.freeze_base,
        tie_alpha_per_rank=cfg.tie_alpha_per_rank,
        local_reparam=cfg.local_reparam,
        target_module_names=["layer3","layer4","fc"],   # wrap all Conv2d + Linear
        include_conv=True,
        include_linear=True,
    )
    return model


def run_one(variant: str, cfg: Config, tr, te):
    device = torch.device(cfg.device)
    model = build_pretrained_resnet18_with_adapters(variant, cfg).to(device)
    opt = make_optimizer(model, lr=cfg.lr, weight_decay=0.0)

    N = len(tr.dataset)
    print(f"\n=== {variant.upper()} run (pretrained base) ===")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss, ce, kl, beta, do_sample = train_epoch(model, tr, opt, device, cfg, epoch, variant, N)

        # mean prediction
        set_bayeslora_sampling(model, False)
        acc_mean = accuracy(model, te, device, sample=False)

        # optional MC prediction (BayesLoRA)
        if variant == "vlora":
            set_bayeslora_sampling(model, True)
            acc_mc = accuracy(model, te, device, sample=True)
        else:
            acc_mc = acc_mean

        print(
            f"Epoch {epoch:02d} | loss {loss:.4f} | ce {ce:.4f} | "
            f"{('kl '+str(round(kl,4))+' | ') if variant=='vlora' else ''}"
            f"β {beta:.2f} | sample {int(do_sample)} | "
            f"acc_mean {acc_mean*100:.2f}% | acc_mc {acc_mc*100:.2f}%"
        )

        # prune-as-you-go for BayesLoRA
        if variant == "vlora":
            pruned, (before, after) = prune_as_you_go(model, cfg, epoch)
            if pruned:
                print(f"[Prune] total ranks: {before} -> {after}")
                # optimizer must be rebuilt after structural prune (new Parameter tensors)
                opt = make_optimizer(model, lr=cfg.lr, weight_decay=0.0)

    t1 = time.time()
    print(f"[{variant.upper()}] total time: {(t1 - t0)/60:.1f} min")

    # final evals
    set_bayeslora_sampling(model, False)
    acc_mean = accuracy(model, te, device, sample=False)
    if variant == "vlora":
        set_bayeslora_sampling(model, True)
        acc_mc = accuracy(model, te, device, sample=True)
    else:
        acc_mc = acc_mean

    print(f"[{variant.upper()}] Final acc (mean): {acc_mean*100:.2f}% | (MC): {acc_mc*100:.2f}%")
    return {"acc_mean": acc_mean, "acc_mc": acc_mc, "model": model}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--freeze-base", action="store_true", default=True)
    p.add_argument("--no-tie-per-rank", action="store_true", default=False)
    p.add_argument("--local-reparam", action="store_true", default=False)

    # BayesLoRA schedules
    p.add_argument("--kl-scale", type=float, default=0.1)
    p.add_argument("--kl-freeze-epochs", type=int, default=3)
    p.add_argument("--beta-warmup-epochs", type=int, default=5)
    p.add_argument("--sample-warmup-epochs", type=int, default=3)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)

    # prune-as-you-go
    p.add_argument("--prune-every", type=int, default=0)
    p.add_argument("--prune-start-epoch", type=int, default=None)
    p.add_argument("--min-ranks", type=int, default=1)

    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        freeze_base=args.freeze_base,
        tie_alpha_per_rank=not args.no_tie_per_rank,
        local_reparam=args.local_reparam,
        kl_scale=args.kl_scale,
        kl_freeze_epochs=args.kl_freeze_epochs,
        beta_warmup_epochs=args.beta_warmup_epochs,
        sample_warmup_epochs=args.sample_warmup_epochs,
        logalpha_thresh=args.logalpha_thresh,
        prune_every=args.prune_every,
        prune_start_epoch=args.prune_start_epoch,
        min_ranks=args.min_ranks,
    )

    torch.manual_seed(cfg.seed)
    tr, te = make_loaders(cfg.batch_size)

    # BayesLoRA (variational)
    res_v = run_one("vlora", cfg, tr, te)

    # Vanilla LoRA
    res_b = run_one("lora", cfg, tr, te)

    print("\n=== Summary ===")
    print(f"BayesLoRA acc (mean): {res_v['acc_mean']*100:.2f}% | MC: {res_v['acc_mc']*100:.2f}%")
    print(f"LoRA     acc (mean): {res_b['acc_mean']*100:.2f}% | MC: {res_b['acc_mc']*100:.2f}%")

if __name__ == "__main__":
    main()