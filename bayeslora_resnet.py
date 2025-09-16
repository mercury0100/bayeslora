#!/usr/bin/env python3
# =============================================
# FILE: bayeslora_resnet.py  (single-file module + CLI)
# =============================================
from __future__ import annotations
import os, csv, json, time, math, argparse, re
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# External helpers expected from your bayeslora package
from bayeslora import apply_lora_adapters, accuracy, make_optimizer, default_prune_start_epoch

# ---------- constants ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------- config & results ----------
@dataclass
class Config:
    # training
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    # lora / adapters
    rank: int = 8
    lora_alpha: Optional[float] = None
    freeze_base: bool = True
    tie_alpha_per_rank: bool = True   # tie within an adapter (A/B for the same rank) — NOT across adapters
    local_reparam: bool = False       # recommend True for BayesLoRA; ignored for plain LoRA
    # bayesian / kl
    kl_scale: float = 0.05
    kl_freeze_epochs: int = 5
    beta_warmup_epochs: int = 10
    sample_warmup_epochs: int = 3
    logalpha_thresh: float = 3.0
    # pruning
    prune_every: int = 0
    prune_start_epoch: Optional[int] = None
    min_ranks: int = 1
    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    # data
    dataset: str = "cifar10"  # or "cifar100"

@dataclass
class EpochLog:
    epoch: int
    loss: float
    ce: float
    kl: float
    beta: float
    do_sample: bool
    acc_mean: float
    acc_mc: float
    ranks_before: Optional[int] = None   # total effective rank (sum across adapters) before prune event
    ranks_after: Optional[int] = None    # total effective rank after prune event

@dataclass
class ExperimentResult:
    acc_mean: float
    acc_mc: float
    ece_mean: float
    ece_mc: float
    nll_mean: float
    nll_mc: float
    brier_mean: float
    brier_mc: float
    model: nn.Module
    logs: Tuple[EpochLog, ...]

# ---------- utils ----------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv_row(csv_path: str, row: Dict[str, object]):
    ensure_dir(os.path.dirname(csv_path) or ".")
    write_header = (not os.path.exists(csv_path)) or os.stat(csv_path).st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_checkpoint(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer,
                    cfg: Config, epoch: int, metrics: Dict[str, float]):
    ensure_dir(os.path.dirname(ckpt_path) or ".")
    payload = {
        "cfg": asdict(cfg),
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "metrics": metrics,
    }
    torch.save(payload, ckpt_path)

# ---------- improved adapter discovery & accounting ----------
# Suffix tokens that typically indicate adapter params
_ADAPTER_TOKENS = [
    "lora_A", "lora_B", "lora_down", "lora_up",
    "A", "B",
    "A_mu", "B_mu", "A_logvar", "B_logvar",
    "log_alpha", "logalpha",
]

# e.g. "layer3.1.conv2.lora_A.weight" -> owner="layer3.1.conv2", suffix="lora_A"
_ADAPTER_RE = re.compile(r"^(?P<owner>.+?)\.(?P<suffix>(" + "|".join(_ADAPTER_TOKENS) + r"))(\.|$)", re.IGNORECASE)

def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    try:
        return model.get_submodule(path)  # torch>=1.12
    except Exception:
        mod = model
        for part in path.split("."):
            mod = getattr(mod, part)
        return mod

def _discover_adapters(model: nn.Module):
    """
    Return a dict owner -> info where info has:
      - 'owner' (str)
      - 'param_names' (set of names belonging to this owner)
      - 'r_attr' (int or None): from module attributes effective_r/active_r/r
      - 'r_inferred' (int or None): inferred from A/B param shapes if attributes missing
    """
    owners: Dict[str, Dict[str, object]] = {}
    name_to_param = dict(model.named_parameters())

    # 1) group adapter params by owner prefix
    for name in name_to_param.keys():
        m = _ADAPTER_RE.match(name)
        if m:
            owner = m.group("owner")
            owners.setdefault(owner, {"owner": owner, "param_names": set()})
            owners[owner]["param_names"].add(name)

    # 2) read rank attributes & infer rank from shapes
    for owner, info in owners.items():
        r_attr = None
        try:
            mod = _get_submodule(model, owner)
        except Exception:
            mod = None
        if mod is not None:
            for key in ("effective_r", "active_r", "r"):
                if hasattr(mod, key):
                    try:
                        r_val = int(getattr(mod, key))
                        if r_val >= 0:
                            r_attr = r_val
                            break
                    except Exception:
                        pass
        owners[owner]["r_attr"] = r_attr

        r_inf = None
        if r_attr is None:
            # try to infer from lora_A/lora_B shapes (r usually the shared small dim)
            A = None; B = None
            for pname in info["param_names"]:
                if pname.endswith(("lora_A.weight", "lora_A", ".A", ".A.weight")):
                    A = name_to_param[pname]
                if pname.endswith(("lora_B.weight", "lora_B", ".B", ".B.weight")):
                    B = name_to_param[pname]
            dims = []
            if A is not None: dims.extend(list(A.shape))
            if B is not None: dims.extend(list(B.shape))
            if dims:
                smalls = [int(d) for d in dims if int(d) <= 2048]
                if smalls:
                    from collections import Counter
                    r_inf = Counter(smalls).most_common(1)[0][0]
                else:
                    r_inf = min(int(d) for d in dims)
        owners[owner]["r_inferred"] = r_inf
    return owners

def total_effective_rank(model: nn.Module) -> int:
    """Sum of effective ranks across discovered adapters (independent across adapters)."""
    owners = _discover_adapters(model)
    tot = 0
    for info in owners.values():
        r = info.get("r_attr")
        if r is None:
            r = info.get("r_inferred")
        if isinstance(r, int) and r >= 0:
            tot += r
    if tot == 0:
        # fallback to attribute scan
        for m in model.modules():
            if hasattr(m, "r"):
                try:
                    r = int(getattr(m, "r"))
                    if r > 0:
                        tot += r
                except Exception:
                    pass
    return int(tot)

def num_adapters(model: nn.Module) -> int:
    """Count distinct adapter owners based on parameter name grouping."""
    owners = _discover_adapters(model)
    if len(owners) > 0:
        return len(owners)
    # fallback to attribute-based
    seen = 0
    for m in model.modules():
        if hasattr(m, "r"):
            try:
                int(getattr(m, "r"))
                seen += 1
            except Exception:
                pass
    return seen

def count_adapter_params(model: nn.Module) -> int:
    """
    Robust adapter-param counter.
    Prefer summing parameters that belong to discovered adapter owners.
    If none discovered, fall back to name-patterns, then theoretical estimate from r & shapes.
    """
    name_to_param = dict(model.named_parameters())
    owners = _discover_adapters(model)
    if owners:
        total = 0
        for info in owners.values():
            for pname in info["param_names"]:
                total += name_to_param[pname].numel()
        if total > 0:
            return total

    # fallback by name patterns
    keys = [
        "lora_", "loraa", "lorab", "lora_down", "lora_up",
        "adapter", ".A.", ".B.",
        "a_mu", "b_mu", "a_logvar", "b_logvar",
        "log_alpha", "logalpha",
    ]
    total = 0
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in keys):
            total += p.numel()
    if total > 0:
        return total

    # last resort: theoretical estimate from r and weight shapes
    total = 0
    for m in model.modules():
        r = getattr(m, "r", None)
        if r is None or int(r) <= 0:
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter):
            if w.dim() == 2:
                out, in_f = w.shape
                total += int(r) * (in_f + out)
            elif w.dim() == 4:
                out, in_c, kh, kw = w.shape
                total += int(r) * (in_c * kh * kw + out)
            else:
                side = int(math.sqrt(w.numel()))
                total += int(r) * (side + side)
        else:
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                total += int(r) * (int(m.in_features) + int(m.out_features))
            elif isinstance(m, nn.Conv2d):
                kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                total += int(r) * (int(m.in_channels) * kh * kw + int(m.out_channels))
    return total

# ---------- data & model ----------
def load_resnet18_pretrained():
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = getattr(ResNet18_Weights, "IMAGENET1K_V1", ResNet18_Weights.DEFAULT)
        return resnet18(weights=weights)
    except Exception:
        try:
            from torchvision.models import resnet18
            return resnet18(weights="IMAGENET1K_V1")
        except Exception:
            from torchvision.models import resnet18
            return resnet18(pretrained=True)

def make_loaders(dataset: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
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
    root = "./data"
    if dataset.lower() == "cifar100":
        train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tf_train)
        test  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tf_test)
    else:
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)
        test  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)
    tr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return tr, te

def set_bayeslora_sampling(model: nn.Module, flag: bool):
    for m in model.modules():
        if hasattr(m, "set_sample"):
            m.set_sample(flag)

def _softmax_probs(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    logits = model(xb)
    return F.softmax(logits, dim=1)

# ---------- SAFE EVAL HELPERS ----------
def eval_top1(model: nn.Module, loader: DataLoader, device: torch.device, sample: bool) -> float:
    """
    Always compute accuracy in eval() with no_grad(), and explicitly set BayesLoRA sampling.
    Restores the previous training mode on exit.
    """
    was_training = model.training
    try:
        model.eval()
        set_bayeslora_sampling(model, sample)
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        return correct / max(1, total)
    finally:
        # turn sampling off by default after eval; restore train/eval mode
        set_bayeslora_sampling(model, False)
        if was_training:
            model.train()
        else:
            model.eval()

def calibration_metrics(model: nn.Module, loader: DataLoader, device: torch.device,
                        n_bins: int = 15, mc_samples: int = 1) -> Tuple[float, float, float]:
    """
    Return (ECE, NLL, Brier).
    If mc_samples>1, we explicitly turn sampling ON and average probabilities across MC draws.
    If mc_samples==1, we turn sampling OFF.
    Evaluates in eval() with no_grad() and restores previous state on exit.
    """
    was_training = model.training
    try:
        model.eval()
        set_bayeslora_sampling(model, mc_samples > 1)

        probs_list, labels_list = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if mc_samples == 1:
                    probs = _softmax_probs(model, xb)
                else:
                    ps = []
                    for _ in range(mc_samples):
                        ps.append(_softmax_probs(model, xb))
                    probs = torch.stack(ps, dim=0).mean(0)
                probs_list.append(probs.cpu()); labels_list.append(yb.cpu())

        probs = torch.cat(probs_list)
        labels = torch.cat(labels_list)

        # NLL
        nll = F.nll_loss((probs + 1e-12).log(), labels, reduction='mean').item()
        # Brier
        one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
        brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()
        # ECE
        conf, preds = probs.max(1); accs = preds.eq(labels)
        bins = torch.linspace(0, 1, n_bins + 1)
        ece = torch.zeros(1)
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                acc_bin = accs[mask].float().mean()
                conf_bin = conf[mask].mean()
                ece += (mask.float().mean() * (conf_bin - acc_bin).abs())
        return float(ece.item()), float(nll), float(brier)
    finally:
        set_bayeslora_sampling(model, False)
        if was_training:
            model.train()
        else:
            model.eval()

# ---------- model build & pruning ----------
def _collect_target_module_names_for_resnet18(model: nn.Module) -> List[str]:
    """
    Enumerate all Conv2d/Linear modules under layer3, layer4, and fc.
    This avoids relying on helper-internal matching semantics.
    """
    targets: List[str] = []
    for name, mod in model.named_modules():
        if not (name.startswith("layer3") or name.startswith("layer4") or name == "fc"):
            continue
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            targets.append(name)
    # Ensure 'fc' is included if present
    if "fc" not in targets and hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        targets.append("fc")
    return sorted(set(targets))

def build_resnet18_with_adapters(variant: str, cfg: Config, num_classes: int) -> nn.Module:
    model = load_resnet18_pretrained()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # >>> Explicitly collect targets under layer3/layer4/fc
    target_names = _collect_target_module_names_for_resnet18(model)
    if len(target_names) == 0:
        # ultra-safe fallback: just fc
        target_names = ["fc"]

    # Apply adapters to the exact modules we enumerated
    model = apply_lora_adapters(
        model,
        variant=variant,
        rank=cfg.rank,
        lora_alpha=(cfg.lora_alpha if cfg.lora_alpha is not None else cfg.rank),
        freeze_base=cfg.freeze_base,
        tie_alpha_per_rank=cfg.tie_alpha_per_rank,   # ties within an adapter only
        local_reparam=cfg.local_reparam,
        target_module_names=target_names,            # <<< explicit list
        include_conv=True,
        include_linear=True,
    )
    return model

def prune_as_you_go(model: nn.Module, cfg: Config, epoch: int) -> Tuple[bool, Tuple[int, int]]:
    """
    Prune ranks independently in each adapter by calling each module's .prune(...).
    Returns (pruned_flag, (total_rank_before, total_rank_after)).
    """
    if cfg.prune_every <= 0:
        return False, (0, 0)
    start = cfg.prune_start_epoch if cfg.prune_start_epoch is not None else default_prune_start_epoch(
        cfg.kl_freeze_epochs, cfg.beta_warmup_epochs
    )
    if epoch < start or (epoch % cfg.prune_every) != 0:
        return False, (0, 0)

    before_total = total_effective_rank(model)
    for m in model.modules():
        if hasattr(m, "prune"):
            try:
                m.prune(log_alpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
            except TypeError:
                try:
                    m.prune(cfg.logalpha_thresh, cfg.min_ranks)
                except Exception:
                    pass
    after_total = total_effective_rank(model)
    return True, (before_total, after_total)

# ---------- training / evaluation ----------
def _compute_beta(cfg: Config, variant: str, epoch: int) -> float:
    if variant != "vlora":
        return 0.0
    if epoch <= cfg.kl_freeze_epochs:
        return 0.0
    t = epoch - cfg.kl_freeze_epochs
    return min(1.0, t / max(1, cfg.beta_warmup_epochs))

def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                device: torch.device, cfg: Config, epoch: int, variant: str, N: int) -> Tuple[float, float, float, float, bool]:
    model.train()
    do_sample = (variant == "vlora") and (epoch > cfg.sample_warmup_epochs)
    set_bayeslora_sampling(model, do_sample)

    beta = _compute_beta(cfg, variant, epoch)
    tot_loss = tot_ce = tot_kl = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        ce = F.cross_entropy(logits, yb)
        if variant == "vlora":
            kl_sum = 0.0
            for m in model.modules():
                if hasattr(m, "kl") and callable(getattr(m, "kl")):
                    kl_sum = kl_sum + m.kl()
            kl_term = cfg.kl_scale * (kl_sum / N)
            loss = ce + beta * kl_term
            tot_kl += float((kl_sum / N).item())
        else:
            loss = ce
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot_loss += float(loss.item()); tot_ce += float(ce.item())
    return tot_loss / len(loader), tot_ce / len(loader), tot_kl, beta, do_sample

def evaluate(model: nn.Module, te: DataLoader, device: torch.device, variant: str) -> Dict[str, float]:
    # Use safe eval helpers for consistent mode/sampling behavior
    acc_mean = eval_top1(model, te, device, sample=False)
    if variant == "vlora":
        acc_mc = eval_top1(model, te, device, sample=True)
        ece_mc, nll_mc, brier_mc = calibration_metrics(model, te, device, mc_samples=5)
    else:
        acc_mc = acc_mean
        ece_mc, nll_mc, brier_mc = None, None, None

    ece_mean, nll_mean, brier_mean = calibration_metrics(model, te, device, mc_samples=1)

    if ece_mc is None:
        ece_mc, nll_mc, brier_mc = ece_mean, nll_mean, brier_mean

    return {
        "acc_mean": float(acc_mean),
        "acc_mc": float(acc_mc),
        "ece_mean": float(ece_mean),
        "ece_mc": float(ece_mc),
        "nll_mean": float(nll_mean),
        "nll_mc": float(nll_mc),
        "brier_mean": float(brier_mean),
        "brier_mc": float(brier_mc),
    }

def run_one(variant: str, cfg: Config, outdir: Optional[str] = None, tag: str = "") -> ExperimentResult:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # data & model
    tr, te = make_loaders(cfg.dataset, cfg.batch_size)
    num_classes = 100 if cfg.dataset.lower() == "cifar100" else 10
    model = build_resnet18_with_adapters(variant, cfg, num_classes).to(device)

    opt = make_optimizer(model, lr=cfg.lr, weight_decay=0.0)
    N = len(tr.dataset)

    # logging / checkpoints
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.dataset}_{variant}_r{cfg.rank}_seed{cfg.seed}{('_'+tag) if tag else ''}_{timestamp}"
    if outdir:
        run_dir = os.path.join(outdir, run_name)
        ensure_dir(run_dir)
        epoch_csv = os.path.join(run_dir, "epochs.csv")
        final_csv = os.path.join(run_dir, "final.csv")
        cfg_json = os.path.join(run_dir, "config.json")
        with open(cfg_json, "w") as f:
            json.dump(asdict(cfg), f, indent=2)
    else:
        run_dir = None
        epoch_csv = final_csv = cfg_json = None

    def log_epoch_row(ep_row: Dict[str, object]):
        if epoch_csv:
            write_csv_row(epoch_csv, ep_row)

    best_acc = -math.inf
    best_ece = math.inf
    logs = []

    print(f"=== {variant.upper()} run ({cfg.dataset}, pretrained base) ===")
    for epoch in range(1, cfg.epochs + 1):
        # ---- 1) Train ----
        loss, ce, kl, beta, do_sample = train_epoch(model, tr, opt, device, cfg, epoch, variant, N)

        # ---- 2) Prune BEFORE eval so metrics reflect the carried-forward model ----
        ranks_before = ranks_after = None
        if variant == "vlora":
            pruned, (before_total, after_total) = prune_as_you_go(model, cfg, epoch)
            if pruned:
                ranks_before, ranks_after = before_total, after_total
                print(f"[Prune] Epoch {epoch:02d}: total effective rank {before_total}->{after_total}")
                # Rebuild optimizer to drop params that are now frozen/removed
                opt = make_optimizer(model, lr=cfg.lr, weight_decay=0.0)

        # ---- 3) Eval AFTER prune (or no-op if nothing pruned) ----
        acc_mean = eval_top1(model, te, device, sample=False)
        acc_mc   = eval_top1(model, te, device, sample=(variant == "vlora"))

        # ---- 4) Accounting & logging ----
        adapter_params   = count_adapter_params(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fc_params        = sum(p.numel() for p in model.fc.parameters() if hasattr(model, "fc") and p.requires_grad) if hasattr(model, "fc") else 0
        total_rank_now   = total_effective_rank(model)
        n_adapters       = num_adapters(model)

        ep_row = {
            "run": run_name,
            "epoch": epoch,
            "loss": round(loss, 6),
            "ce": round(ce, 6),
            "kl": round(kl, 6),
            "beta": round(beta, 6),
            "do_sample": int(do_sample),
            "acc_mean": round(float(acc_mean), 6),
            "acc_mc": round(float(acc_mc), 6),
            "ranks_before": ranks_before if ranks_before is not None else "",
            "ranks_after": ranks_after if ranks_after is not None else "",
            "total_rank": total_rank_now,      # total effective rank (sum across adapters)
            "num_adapters": n_adapters,
            "adapter_params": adapter_params,
            "trainable_params": trainable_params,
            "fc_params": fc_params,
        }
        if epoch_csv:
            write_csv_row(epoch_csv, ep_row)

        print(
            f"Epoch {epoch:02d} | loss {loss:.4f} | ce {ce:.4f} | "
            f"{('kl ' + str(round(kl, 4)) + ' | ') if variant == 'vlora' else ''}"
            f"β {beta:.2f} | sample {int(do_sample)} | "
            f"acc_mean {acc_mean*100:.2f}% | acc_mc {acc_mc*100:.2f}% | "
            f"adapters {adapter_params} | total_rank {total_rank_now} / {n_adapters} adapters"
        )
        logs.append(EpochLog(epoch, loss, ce, kl, beta, do_sample, float(acc_mean), float(acc_mc),
                             ranks_before, ranks_after))

        # ---- 5) Checkpoints (use post-prune metrics) ----
        if run_dir:
            metrics_epoch = {"acc_mean": float(acc_mean), "acc_mc": float(acc_mc), "epoch": epoch,
                             "total_rank": total_rank_now}
            # best-acc
            if float(acc_mean) > best_acc:
                best_acc = float(acc_mean)
                save_checkpoint(os.path.join(run_dir, "best_acc.pt"), model, opt, cfg, epoch, metrics_epoch)
            # best-calibration (mean predictive)
            ece_mean_q, _, _ = calibration_metrics(model, te, device, mc_samples=1)
            if float(ece_mean_q) < best_ece:
                best_ece = float(ece_mean_q)
                save_checkpoint(os.path.join(run_dir, "best_ece.pt"), model, opt, cfg, epoch,
                                {**metrics_epoch, "ece_mean": ece_mean_q})
            # periodic
            save_checkpoint(os.path.join(run_dir, f"epoch_{epoch:03d}.pt"), model, opt, cfg, epoch, metrics_epoch)

    # final eval with calib metrics (safe & consistent)
    metrics = evaluate(model, te, device, variant)
    total_rank_final = total_effective_rank(model)

    print(
        f"[{variant.upper()}] Final acc (mean): {metrics['acc_mean']*100:.2f}% | (MC): {metrics['acc_mc']*100:.2f}% | "
        f"ECE (mean): {metrics['ece_mean']:.4f} | ECE (MC): {metrics['ece_mc']:.4f} | "
        f"NLL (mean): {metrics['nll_mean']:.4f} | NLL (MC): {metrics['nll_mc']:.4f} | "
        f"Brier (mean): {metrics['brier_mean']:.4f} | Brier (MC): {metrics['brier_mc']:.4f} | "
        f"Total effective rank: {total_rank_final}"
    )

    # final CSV
    if final_csv:
        row = {
            "run": run_name,
            **asdict(cfg),
            "variant": variant,
            **{k: round(v, 8) for k, v in metrics.items()},
            "adapter_params_final": count_adapter_params(model),
            "total_rank_final": total_rank_final,   # total effective rank at the end
        }
        write_csv_row(final_csv, row)

    return ExperimentResult(
        acc_mean=metrics["acc_mean"], acc_mc=metrics["acc_mc"],
        ece_mean=metrics["ece_mean"], ece_mc=metrics["ece_mc"],
        nll_mean=metrics["nll_mean"], nll_mc=metrics["nll_mc"],
        brier_mean=metrics["brier_mean"], brier_mc=metrics["brier_mc"],
        model=model, logs=tuple(logs)
    )

# Convenience: simple factory to map friendly names to variants
VARIANT_MAP = {
    "lora": "lora",
    "bayeslora": "vlora",
    "vlora": "vlora",
}

__all__ = [
    "Config", "ExperimentResult", "EpochLog",
    "run_one", "build_resnet18_with_adapters", "make_loaders",
    "calibration_metrics", "prune_as_you_go", "VARIANT_MAP",
]

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="BayesLoRA/LoRA ResNet18 on CIFAR with CSV logging & checkpoints")
    # general
    p.add_argument("--method", type=str, default="bayeslora", choices=["lora", "bayeslora", "vlora"], help="Which adapter method to run")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--freeze-base", action="store_true", default=True)
    p.add_argument("--no-tie-per-rank", action="store_true", default=False)
    p.add_argument("--local-reparam", action="store_true", default=False)
    p.add_argument("--kl-scale", type=float, default=0.05)
    p.add_argument("--kl-freeze-epochs", type=int, default=5)
    p.add_argument("--beta-warmup-epochs", type=int, default=10)
    p.add_argument("--sample-warmup-epochs", type=int, default=3)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)
    p.add_argument("--prune-every", type=int, default=0)
    p.add_argument("--prune-start-epoch", type=int, default=None)
    p.add_argument("--min-ranks", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    # logging / ckpts
    p.add_argument("--outdir", type=str, default="runs", help="Directory to write CSV/ckpts (per-run subfolder)")
    p.add_argument("--tag", type=str, default="", help="Optional tag for the run name")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        rank=args.rank, lora_alpha=args.lora_alpha,
        freeze_base=args.freeze_base, tie_alpha_per_rank=not args.no_tie_per_rank,
        local_reparam=args.local_reparam, kl_scale=args.kl_scale,
        kl_freeze_epochs=args.kl_freeze_epochs, beta_warmup_epochs=args.beta_warmup_epochs,
        sample_warmup_epochs=args.sample_warmup_epochs, logalpha_thresh=args.logalpha_thresh,
        prune_every=args.prune_every, prune_start_epoch=args.prune_start_epoch, min_ranks=args.min_ranks,
        seed=args.seed, dataset=args.dataset,
    )
    variant = VARIANT_MAP[args.method]
    res = run_one(variant, cfg, outdir=args.outdir, tag=args.tag)

    # brief final echo
    print("=== Final Summary ===")
    print({k: v for k, v in asdict(cfg).items() if k not in ("device",)})
    print({
        "acc_mean": res.acc_mean, "acc_mc": res.acc_mc,
        "ece_mean": res.ece_mean, "ece_mc": res.ece_mc,
        "nll_mean": res.nll_mean, "nll_mc": res.nll_mc,
        "brier_mean": res.brier_mean, "brier_mc": res.brier_mc,
    })

if __name__ == "__main__":
    main()