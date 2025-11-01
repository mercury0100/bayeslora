#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeBERTa GLUE: BayesLoRA rank dynamics + pruning schedule + head-to-head vs AdaLoRA and LoRA.

Example:
python run_glue_bench.py \
  --tasks sst2 \
  --seeds 1 \
  --kl_scale 1e-6 \
  --log_alpha_threshold 4.0 \
  --epochs 2
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_JIT", "0")

import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from datasets import load_dataset
import evaluate

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# BayesLoRA & LoRA wrappers (your hf.py)
from bayeslora.hf import (
    apply_bayeslora,
    bayeslora_step,
    bayeslora_prune,
    get_adapter_ranks,
    count_adapter_params,
    set_bayeslora_sampling,
    LoRAWrapper,
)

from peft import get_peft_model, AdaLoraConfig, TaskType
from peft.utils import get_peft_model_state_dict  # robust save/load for PEFT adapters

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLUE_TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
}
GLUE_NUM_LABELS = {"sst2":2,"mnli":3,"qnli":2,"rte":2,"cola":2,"mrpc":2,"qqp":2}


def default_targets_for_deberta() -> List[str]:
    return [
        "attention.self.query_proj", "attention.self.key_proj", "attention.self.value_proj",
        "attention.output.dense",
        "intermediate.dense", "output.dense",
    ]


def is_classifier_param(name: str) -> bool:
    if ".classifier." in name:
        return True
    if name.endswith("classifier.weight") or name.endswith("classifier.bias"):
        return True
    if ".score." in name or ".scores." in name:
        return True
    return False


def preprocess_batch(raw_batch, tokenizer, task, max_len):
    key1, key2 = GLUE_TASK_TO_KEYS[task]
    if isinstance(raw_batch, dict):
        if key2 is None:
            enc = tokenizer(raw_batch[key1], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        else:
            enc = tokenizer(raw_batch[key1], raw_batch[key2], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        labels = torch.tensor(raw_batch["label"], dtype=torch.long)
    else:
        if key2 is None:
            texts1 = [ex[key1] for ex in raw_batch]
            enc = tokenizer(texts1, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        else:
            texts1 = [ex[key1] for ex in raw_batch]
            texts2 = [ex[key2] for ex in raw_batch]
            enc = tokenizer(texts1, texts2, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        labels = torch.tensor([ex["label"] for ex in raw_batch], dtype=torch.long)
    enc["labels"] = labels
    return enc


def build_datasets(task: str, cache_dir: Optional[str] = None):
    ds = load_dataset("glue", task, cache_dir=cache_dir)
    if os.environ.get("FAST_DEV_RUN", "0") == "1":
        train = ds["train"].shuffle(seed=123).select(range(512))
        val   = (ds["validation_matched"] if task=="mnli" else ds["validation"]).shuffle(seed=123).select(range(256))
    else:
        train = ds["train"]
        val   = ds["validation_matched"] if task=="mnli" else ds["validation"]
    return train, val


def metric_accuracy():
    return evaluate.load("accuracy")

def ema(prev, x, beta=0.9):
    return beta * prev + (1 - beta) * x if prev is not None else x


def save_json(o, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(o, f, indent=2)


def module_total_rank(ranks: Dict[str,int]) -> int:
    return int(sum(ranks.values()))


def uniform_rank_allocation(target_names: List[str], total_rank: int) -> Dict[str,int]:
    if len(target_names) == 0 or total_rank <= 0:
        return {name: 0 for name in target_names}
    base = total_rank // len(target_names)
    extra = total_rank % len(target_names)
    out = {name: base for name in target_names}
    for i, name in enumerate(sorted(target_names)[:extra]):
        out[name] += 1
    return out


def iter_minibatches(dataset, batch_size: int, shuffle: bool, seed: Optional[int]) -> Iterable[List[dict]]:
    idxs = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    for i in range(0, len(dataset), batch_size):
        yield [dataset[j] for j in idxs[i:i+batch_size]]

# -------------------------
# AdaLoRA telemetry (STATE-DICT ONLY; PEFT 0.17.x SVDLinear-safe)
# -------------------------
def _parse_sdict_for_ranks(state_dict_like):
    """
    Robust parser for PEFT 0.17.x AdaLoRA:
      - Accepts keys that end with `.lora_A`, `.lora_B`, `.lora_E`, `.lora_ind`
      - Accepts keys with or without an explicit adapter segment
      - Effective rank priority: lora_ind (sum>0) > lora_E (nonzero rows) > min(A_rows, B_dim)
    """
    A_rows, B_dim = {}, {}
    E_rows_nz, IND_nz = {}, {}

    def split_key(key: str, tag: str):
        if f".{tag}." in key:
            modpath, rest = key.split(f".{tag}.", 1)
            adapter = rest.split(".", 1)[0] if "." in rest else "default"
        elif key.endswith(f".{tag}"):
            modpath, adapter = key[:-(len(tag)+1)], "default"
        else:
            return None, None
        return modpath, adapter

    for k, v in state_dict_like.items():
        if not isinstance(v, torch.Tensor):
            continue

        mod, ad = split_key(k, "lora_ind")
        if mod is not None:
            IND_nz[(mod, ad)] = int((v != 0).sum().item())
            continue

        mod, ad = split_key(k, "lora_E")
        if mod is not None:
            if v.ndim == 2:
                E_rows_nz[(mod, ad)] = int((v.abs().max(dim=1).values > 0).sum().item())
            else:
                E_rows_nz[(mod, ad)] = int((v != 0).sum().item())
            continue

        mod, ad = split_key(k, "lora_A")
        if mod is not None:
            A_rows[(mod, ad)] = int(v.shape[0]) if v.ndim == 2 else 0
            continue

        mod, ad = split_key(k, "lora_B")
        if mod is not None:
            B_dim[(mod, ad)] = int(min(v.shape)) if v.ndim == 2 else 0
            continue

    per, total = {}, 0
    keys = set().union(A_rows.keys(), B_dim.keys(), E_rows_nz.keys(), IND_nz.keys())
    for mk in keys:
        r_ind = IND_nz.get(mk, 0)
        r_e   = E_rows_nz.get(mk, 0)
        rA    = A_rows.get(mk, 0)
        rB    = B_dim.get(mk, 0)
        r_shape = rA if (rB == 0) else (rB if (rA == 0) else min(rA, rB))

        # Trust IND/E only when nonzero; otherwise fall back to shapes
        if r_ind > 0:
            r_eff = r_ind
        elif r_e > 0:
            r_eff = r_e
        else:
            r_eff = r_shape

        per[f"{mk[0]}.lora[{mk[1]}]"] = int(r_eff)
        total += int(r_eff)

    return per, int(total)


def _adalora_ranks_from_state_dict(model) -> Tuple[Dict[str, int], int]:
    # Prefer the PEFT helper, then fall back to the full model SD
    sdict = get_peft_model_state_dict(model)
    if any((".lora_" in k) for k in sdict.keys()):
        return _parse_sdict_for_ranks(sdict)
    full_sd = model.state_dict()
    if any((".lora_" in k) for k in full_sd.keys()):
        return _parse_sdict_for_ranks(full_sd)
    return {}, 0


def audit_peft_ranks(model, adapter_key: Optional[str] = "default") -> Tuple[Dict[str, int], int]:
    per_all, tot_all = _adalora_ranks_from_state_dict(model)
    if adapter_key is None:
        return per_all, tot_all
    per_filt, tot_filt = {}, 0
    needle = f"lora[{adapter_key}]"
    for name, r in per_all.items():
        if name.endswith(needle):
            per_filt[name] = r; tot_filt += r
    return per_filt, tot_filt

# -------------------------
# Unified logging helpers
# -------------------------
def print_step_metrics(task: str, step: int, total_steps: int, ce: float, loss: float, lr: float, extras: Optional[Dict[str, float]] = None, every: int = 1):
    """Consistent per-step line across methods; extras may include KL_scaled, KL_raw, etc."""
    if step % max(1, every) != 0:
        return
    parts = [f"[{task}] step {step}/{total_steps}", f"CE={ce:.6f}", f"Loss={loss:.6f}", f"lr={lr:.3e}"]
    if extras:
        for k, v in extras.items():
            if k == "KL_raw":
                parts.append(f"KL_raw≈{v:.6f}")
            else:
                parts.append(f"{k}={v:.6f}")
    print(" | ".join(parts), flush=True)

def save_rank_log_json(per_mod: dict, step: int, path: Path):
    """Append or create JSON log for rank telemetry."""
    entry = {"step": step, "per_module": per_mod}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.load(open(path))
        except Exception:
            data = []
        data.append(entry)
    else:
        data = [entry]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def log_rank_snapshot(method: str, step: int, per_mod: Dict[str, int], modules_count: int, json_path: Path):
    tot = int(sum(per_mod.values()))
    avg = tot / max(1, modules_count)
    print(f"[{method}@{step}] total_rank={tot} (avg={avg:.4f})", flush=True)
    save_rank_log_json(per_mod, step=step, path=json_path)

def aggregate_rank_logs(paths: List[Path]) -> List[Dict[str, Any]]:
    logs_all = []
    for p in paths:
        if p.exists():
            try:
                logs_all.extend(json.load(open(p)))
            except Exception:
                pass
    return logs_all

# -------------------------
# Plotting
# -------------------------
def plot_rank_dynamics(task: str, logs: List[Dict[str,Any]], out_path: Path, num_layers: Optional[int] = None):
    if not logs:
        return
    steps = sorted({x["step"] for x in logs})
    step_idx = {s: i for i, s in enumerate(steps)}
    T = len(steps)

    if num_layers is None:
        seen_layers = set()
        for x in logs:
            for name in x["per_module"].keys():
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layer" and i + 1 < len(parts):
                        try:
                            seen_layers.add(int(parts[i+1]))
                        except:
                            pass
        num_layers = max(seen_layers) + 1 if seen_layers else 1

    mat = np.full((num_layers, T), np.nan, dtype=float)

    def sum_per_layer(per_module: Dict[str,int]) -> Dict[int,int]:
        layer_sums: Dict[int,int] = {L: 0 for L in range(num_layers)}
        for name, r in per_module.items():
            L = None
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layer" and i + 1 < len(parts):
                    try:
                        L = int(parts[i+1]); break
                    except:
                        L = None
            if L is None or L < 0 or L >= num_layers:
                continue
            layer_sums[L] += int(r)
        return layer_sums

    for x in logs:
        t = step_idx[x["step"]]
        layer_sums = sum_per_layer(x["per_module"])
        for L in range(num_layers):
            mat[L, t] = layer_sums.get(L, 0)

    for L in range(num_layers):
        last = 0.0
        for t in range(T):
            if not np.isfinite(mat[L, t]):
                mat[L, t] = last
            else:
                last = mat[L, t]

    total_ranks = [float(np.nansum(mat[:, t])) for t in range(T)]

    fig = plt.figure(figsize=(11, 5.5))
    ax1 = fig.add_subplot(2,1,1)
    im = ax1.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    ax1.set_ylabel("Encoder layer")
    ax1.set_title(f"{task.upper()} – Sum of adapter ranks per layer over steps")
    xticks = list(range(0, T, max(1, T // 10)))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([steps[i] for i in xticks], rotation=0)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02, label="Sum of ranks")

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(steps, total_ranks, linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Total rank (Σ layers)")
    ax2.set_title("Total adapter rank over time")
    ax2.grid(True, linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_accuracy_over_steps(task: str,
                             acc_logs_by_seed: Dict[int, List[Dict[str,Any]]],
                             out_path: Path,
                             title_suffix: str = ""):
    if not acc_logs_by_seed:
        return
    all_steps = sorted({x["step"] for logs in acc_logs_by_seed.values() for x in logs})
    if not all_steps:
        return
    curves = []
    for _, logs in acc_logs_by_seed.items():
        step_to_val = {x["step"]: x["accuracy"] for x in logs}
        curves.append([step_to_val.get(s, np.nan) for s in all_steps])
    M = np.array(curves, dtype=float)
    mean = np.nanmean(M, axis=0)
    std  = np.nanstd(M, axis=0)

    fig = plt.figure(figsize=(10.5, 4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(all_steps, mean, linewidth=2, label="mean acc")
    ax.fill_between(all_steps, mean-std, mean+std, alpha=0.2, label="±1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ttl = f"{task.upper()} – Accuracy over steps (mean±std across seeds)"
    if title_suffix:
        ttl += f" – {title_suffix}"
    ax.set_title(ttl)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_accuracy_comparison(task: str,
                             bayes: Dict[int, List[Dict[str,Any]]],
                             adalora: Optional[Dict[int, List[Dict[str,Any]]]],
                             lora: Optional[Dict[int, List[Dict[str,Any]]]],
                             out_path: Path,
                             title_suffix: str = ""):
    def mean_curve(acc_logs_by_seed):
        if not acc_logs_by_seed:
            return None, None, None
        steps = sorted({x["step"] for logs in acc_logs_by_seed.values() for x in logs})
        if not steps:
            return None, None, None
        curves = []
        for _, logs in acc_logs_by_seed.items():
            m = {x["step"]: x["accuracy"] for x in logs}
            curves.append([m.get(s, np.nan) for s in steps])
        M = np.array(curves, dtype=float)
        return steps, np.nanmean(M, axis=0), np.nanstd(M, axis=0)

    s_b, m_b, sd_b = mean_curve(bayes)
    s_a, m_a, sd_a = mean_curve(adalora or {})
    s_l, m_l, sd_l = mean_curve(lora or {})

    fig = plt.figure(figsize=(10.5, 4))
    ax = fig.add_subplot(1,1,1)

    if s_b is not None:
        ax.plot(s_b, m_b, linewidth=2, label="BayesLoRA")
        ax.fill_between(s_b, m_b - sd_b, m_b + sd_b, alpha=0.15)
    if s_a is not None:
        ax.plot(s_a, m_a, linewidth=2, linestyle="--", label="AdaLoRA")
    if s_l is not None:
        ax.plot(s_l, m_l, linewidth=2, linestyle=":", label="LoRA (uniform)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ttl = f"{task.upper()} – Accuracy comparison (mean±std)"
    if title_suffix:
        ttl += f" – {title_suffix}"
    ax.set_title(ttl)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -------------------------
# Training / Evaluation
# -------------------------
@dataclass
class TrainConfig:
    model_name: str = "microsoft/deberta-v3-base"
    task: str = "sst2"
    max_len: int = 256
    batch_size: int = 64
    epochs: int = 30
    max_steps: Optional[int] = None
    eval_steps: Optional[int] = None   # if None -> once per epoch
    device: str = "cuda:0"

    # DeBERTa-stable optimisation
    adapter_lr: float = 1e-3
    head_lr: float = 1e-2
    wd: float = 1e-2
    warmup_ratio: float = 0.06
    clip_grad: float = 0.1

    # BayesLoRA
    kl_scale: float = 1e-6
    prune_every: int = 1000
    prune_start: Optional[int] = None
    log_alpha_threshold: float = 3.0
    lora_r_init: int = 8
    lora_alpha: int = 16
    tie_alpha_per_rank: bool = True
    local_reparam: bool = False

    # Misc
    seeds: Tuple[int,...] = (1, 2, 3)
    out_dir: str = "outputs"
    target_modules: Optional[List[str]] = None
    trainable_allowlist: Tuple[str,...] = ("classifier",)


def create_model(num_labels: int, model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)


def evaluate_accuracy(model, dataset, tokenizer, task, max_len, device, batch_size=128) -> float:
    model.eval()
    acc = metric_accuracy()
    for batch in iter_minibatches(dataset, batch_size=batch_size, shuffle=False, seed=1234):
        enc = preprocess_batch(batch, tokenizer, task, max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        # Only toggle BayesLoRA sampling if present
        try:
            set_bayeslora_sampling(model, False)
        except Exception:
            pass
        logits = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc}).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        acc.add_batch(predictions=preds, references=enc["labels"].cpu().numpy())
    return float(acc.compute()["accuracy"])


# ---------- No KL floor (you control kl_scale) ----------
def set_kl_scale(model: nn.Module, new_scale: float):
    for m in model.modules():
        if hasattr(m, "_kl_scale"):
            m._kl_scale = float(new_scale)

def scaled_kl(model: nn.Module) -> torch.Tensor:
    return bayeslora_step(model, 1.0)
# -------------------------------------------------------


def build_param_groups(model: nn.Module, adapter_lr: float, head_lr: float, wd: float):
    """Stable groups: no WD on adapter means/logvars; smaller LR for logvars; WD on classifier weights only."""
    adapter_means, logvars = [], []
    head_decay, head_nodecay = [], []

    def is_logvar(n: str) -> bool:
        return any(k in n for k in ("rank_log_sigma2", "A_log_sigma2", "B_log_sigma2"))
    def is_adapter_mean(n: str) -> bool:
        return any(k in n for k in ("A_mu", "B_mu"))
    def is_head(n: str) -> bool:
        return (".classifier." in n) or n.endswith("classifier.weight") or n.endswith("classifier.bias") \
               or ".score." in n or ".scores." in n
    def is_norm_or_bias(n: str, p: torch.nn.Parameter) -> bool:
        return n.endswith(".bias") or ("LayerNorm" in n) or ("layer_norm" in n) or p.ndim == 1

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_logvar(n):
            logvars.append(p)                         # small LR, no WD
        elif is_adapter_mean(n):
            adapter_means.append(p)                   # adapter means, no WD
        elif is_head(n):
            (head_nodecay if is_norm_or_bias(n, p) else head_decay).append(p)

    groups = []
    if adapter_means:
        groups.append({"params": adapter_means, "lr": adapter_lr, "weight_decay": 0.0})
    if logvars:
        groups.append({"params": logvars, "lr": adapter_lr * 0.1, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": wd})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})
    return groups


def train_bayeslora(cfg: TrainConfig, train, val, model, tokenizer, target_modules: List[str], seed: int, run_dir: Path) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)

    model = apply_bayeslora(
        model, target_modules=target_modules, rank=cfg.lora_r_init,
        lora_alpha=cfg.lora_alpha, kl_scale=cfg.kl_scale,
        tie_alpha_per_rank=cfg.tie_alpha_per_rank,
        local_reparam=cfg.local_reparam, freeze_base=True,
    )
    set_kl_scale(model, cfg.kl_scale)

    allow = tuple(set(cfg.trainable_allowlist or ()))
    for n, p in model.named_parameters():
        if any(tok in n for tok in allow):
            p.requires_grad = True

    # Warm kernels (ensures buffers/keys exist & CUDA kernels are ready)
    k1, k2 = GLUE_TASK_TO_KEYS[cfg.task]
    texts1 = ["hello world"] * 2
    texts2 = ["hello there"] * 2 if k2 is not None else None
    tok = tokenizer(texts1, texts2, truncation=True, padding=True, max_length=cfg.max_len, return_tensors="pt")
    tok = {k: v.to(cfg.device) for k, v in tok.items()}
    with torch.no_grad():
        _ = model(**{k: tok[k] for k in ("input_ids","attention_mask","token_type_ids") if k in tok}).logits
    if device.type == "cuda":
        torch.cuda.synchronize()

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps_default = cfg.epochs * steps_per_epoch
    total_steps = cfg.max_steps or total_steps_default
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    prune_start = cfg.prune_start if (cfg.prune_start is not None) else warmup_steps
    eval_every = cfg.eval_steps or steps_per_epoch  # eval once per epoch by default

    def make_optim_and_sched(step_already: int = 0):
        groups = build_param_groups(model, cfg.adapter_lr, cfg.head_lr, cfg.wd)
        opt = torch.optim.AdamW(groups)
        sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        sch.last_epoch = step_already
        return opt, sch

    optimizer, scheduler = make_optim_and_sched(step_already=0)

    ranks_log, acc_log = [], []
    ema_acc = None
    best_acc = -1.0
    best_ckpt = run_dir / "bayeslora_best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_snapshot = None

    # Init rank telemetry
    r0 = module_total_rank(get_adapter_ranks(model))
    print(f"[bayeslora@init] total_rank={r0}", flush=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sanity] kl_scale={cfg.kl_scale:.3e} | trainable_params={n_trainable/1e6:.3f}M "
          f"| lr(adapter)={cfg.adapter_lr:.3e} | lr(head)={cfg.head_lr:.3e} "
          f"| warmup_steps={warmup_steps} | prune_start={prune_start} | steps/epoch={steps_per_epoch}", flush=True)

    model.train()
    step = 0
    epoch = 0
    while step < total_steps:
        # Shuffle once per epoch
        for batch in iter_minibatches(train, batch_size=cfg.batch_size, shuffle=True, seed=seed + epoch):
            if step >= total_steps:
                break

            enc = preprocess_batch(batch, tokenizer, cfg.task, cfg.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc})

            ce = out.loss
            sk = scaled_kl(model)
            raw_kl = (sk / cfg.kl_scale) if cfg.kl_scale > 0 else torch.tensor(0.0, device=ce.device)
            loss = ce + sk

            lr_now = scheduler.get_last_lr()[0]
            print_step_metrics(
                task=cfg.task, step=step, total_steps=total_steps,
                ce=float(ce), loss=float(loss), lr=lr_now,
                extras={"KL_scaled": float(sk), "KL_raw": float(raw_kl)},
                every=max(1, eval_every // 5)
            )

            # Online pruning AFTER warmup (NO min_ranks)
            if (cfg.prune_every > 0) and (step >= prune_start) and (step % cfg.prune_every == 0):
                pre_total = module_total_rank(get_adapter_ranks(model))
                changed = bayeslora_prune(model, logalpha_thresh=cfg.log_alpha_threshold, min_ranks=0)
                post_ranks = get_adapter_ranks(model)
                post_total = module_total_rank(post_ranks)
                ranks_log.append({"step": step, "total_rank": post_total, "per_module": post_ranks})
                print(f"[prune@{step}] th={cfg.log_alpha_threshold:.2f} | total_rank {pre_total} -> {post_total} | changed={bool(changed)}", flush=True)
                optimizer, scheduler = make_optim_and_sched(step_already=step)

            # Backprop guard
            if any(p.requires_grad for p in model.parameters()):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.clip_grad)
                optimizer.step()
                scheduler.step()

            # Periodic eval
            if (step % eval_every == 0) or (step == total_steps - 1):
                ranks = get_adapter_ranks(model)
                total_rank = module_total_rank(ranks)
                ranks_log.append({"step": step, "total_rank": total_rank, "per_module": ranks})
                acc = evaluate_accuracy(model, val, tokenizer, cfg.task, cfg.max_len, device)
                model.train()
                ema_acc = ema(ema_acc, acc, beta=0.9)
                acc_log.append({"step": step, "accuracy": acc, "ema": ema_acc})
                if acc > best_acc:
                    best_acc = acc
                    best_model_snapshot = copy.deepcopy(model).to("cpu")
                    torch.save(best_model_snapshot.state_dict(), best_ckpt)

            step += 1
        epoch += 1

    # Budget from END-OF-TRAINING model (not the best snapshot)
    end_ranks = get_adapter_ranks(model)
    end_total_rank = module_total_rank(end_ranks)
    adapter_params_end = count_adapter_params(model)

    # Restore best snapshot for downstream eval
    if best_model_snapshot is not None:
        model = best_model_snapshot.to(device)
    else:
        print("[warn] no best snapshot captured; returning current model", flush=True)

    set_kl_scale(model, cfg.kl_scale)

    # Persist logs
    save_json(ranks_log, run_dir / "bayeslora_ranks_log.json")
    save_json(acc_log, run_dir / "bayeslora_accuracy_log.json")

    return {
        "model": model,
        "best_acc": best_acc,
        "ranks_log": ranks_log,
        "acc_log": acc_log,
        "budget_ranks": end_ranks,
        "budget_total_rank": end_total_rank,
        "adapter_params": int(adapter_params_end),
    }


def train_uniform_lora(cfg: TrainConfig, train, val, tokenizer, seed, target_modules, final_total_rank, run_dir: Path, num_labels: int):
    device = torch.device(cfg.device)
    model = create_model(num_labels=num_labels, model_name=cfg.model_name)
    model.to(device)

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    actual_targets = []
    for n, m in model.named_modules():
        if any(t in n for t in target_modules) and isinstance(m, nn.Linear):
            actual_targets.append(n)
    alloc = uniform_rank_allocation(actual_targets, final_total_rank)

    for name, mod in list(model.named_modules()):
        if name in alloc and isinstance(mod, nn.Linear):
            r = alloc[name]
            if r == 0:
                continue
            d_out, d_in = mod.out_features, mod.in_features
            parent = model
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            wrapper = LoRAWrapper(mod, d_in, d_out, r=r, lora_alpha=cfg.lora_alpha)
            setattr(parent, name.split(".")[-1], wrapper)

    # Trainable: LoRA A/B + classifier
    for n, p in model.named_parameters():
        if (".A" in n or ".B" in n) or is_classifier_param(n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    lora_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_classifier_param(n):
            head_params.append(p)
        else:
            lora_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.adapter_lr, "weight_decay": cfg.wd},
         {"params": head_params, "lr": cfg.head_lr,    "weight_decay": cfg.wd}]
    )

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps_default = cfg.epochs * steps_per_epoch
    total_steps = cfg.max_steps or total_steps_default
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    eval_every = cfg.eval_steps or steps_per_epoch

    # Fixed LoRA rank telemetry
    lora_rank_json = run_dir / "lora_ranks_log.json"
    per_mod_fixed = {name: int(alloc[name]) for name in alloc}
    log_rank_snapshot("lora", 0, per_mod_fixed, modules_count=len(per_mod_fixed), json_path=lora_rank_json)

    best_acc = -1.0
    best_ckpt = run_dir / "lora_best.pt"
    acc_log = []
    model.train()
    step = 0
    epoch = 0
    while step < total_steps:
        for batch in iter_minibatches(train, batch_size=cfg.batch_size, shuffle=True, seed=seed + epoch):
            if step >= total_steps:
                break
            enc = preprocess_batch(batch, tokenizer, cfg.task, cfg.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc})
            loss = out.loss

            lr_now = scheduler.get_last_lr()[0]
            print_step_metrics(
                task=cfg.task, step=step, total_steps=total_steps,
                ce=float(loss), loss=float(loss), lr=lr_now,
                extras=None, every=max(1, eval_every // 5)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.clip_grad)
            optimizer.step()
            scheduler.step()

            if (step % eval_every == 0) or (step == total_steps - 1):
                log_rank_snapshot("lora", step, per_mod_fixed, modules_count=len(per_mod_fixed), json_path=lora_rank_json)
                acc = evaluate_accuracy(model, val, tokenizer, cfg.task, cfg.max_len, device)
                model.train()
                acc_log.append({"step": step, "accuracy": acc})
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), best_ckpt)
            step += 1
        epoch += 1
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return model, best_acc, acc_log


# -------------------------
# AdaLoRA helpers
# -------------------------
def _adalora_update(model, global_step_0based: int) -> bool:
    """
    Call AdaLoRA's reallocation regardless of wrapper depth.
    Use 0-based global step and call AFTER backward(), BEFORE optimizer.step().
    Adds guard against empty importance tensors (PEFT edge case).
    """
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for tgt in candidates:
        if tgt is not None and hasattr(tgt, "update_and_allocate"):
            alloc = getattr(tgt, "rankallocator", None)
            if alloc is not None:
                # Check for empty importance tensors (bug trigger)
                scores = getattr(alloc, "scores", None)
                if scores and all(
                    (s is None or (hasattr(s, "numel") and s.numel() == 0))
                    for s in scores.values()
                ):
                    print(f"[adalora@{global_step_0based}] skip empty allocator for {tgt.__class__.__name__}")
                    continue

                # Defensive clamp: ensure budget never drops below 1
                if hasattr(alloc, "rank_budget"):
                    alloc.rank_budget = max(1, getattr(alloc, "rank_budget", 1))

            tgt.update_and_allocate(global_step_0based)
            return True
    return False

# -------------------------
# Main orchestration
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--tasks", type=str, nargs="+", default=["sst2","mnli","qnli","rte","cola","mrpc","qqp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None, help="if None, evaluate once per epoch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--kl_scale", type=float, default=1e-6)  # aligned with TrainConfig
    parser.add_argument("--prune_every", type=int, default=1000)
    parser.add_argument("--prune_start", type=int, default=None, help="first step eligible for pruning; default=warmup_steps")
    parser.add_argument("--log_alpha_threshold", type=float, nargs="+", default=[3.0])
    parser.add_argument("--lora_r_init", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_DATASETS_CACHE", None))
    args = parser.parse_args()

    thresholds = list(args.log_alpha_threshold) if isinstance(args.log_alpha_threshold, (list, tuple)) else [args.log_alpha_threshold]

    # Early CUDA context init
    if args.device.startswith("cuda"):
        _ = torch.randn(1, device=args.device)
        torch.cuda.synchronize()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir)
    results_csv = out_root / "results" / f"{timestamp}_glue_bayeslora_adalora_lora.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        print(f"\n=== Task: {task} ===", flush=True)
        num_labels = GLUE_NUM_LABELS[task]
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        train, val = build_datasets(task, cache_dir=args.cache_dir)
        target_modules = default_targets_for_deberta()
        num_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers

        for th in thresholds:
            print(f"\n--- Threshold ablation: th={th:.2f} ---", flush=True)

            # BayesLoRA logs
            bayes_acc_logs_by_seed: Dict[int, List[Dict[str,Any]]] = {}
            bayes_rows = []
            end_ranks_per_seed = []

            # ---- BayesLoRA training per seed ----
            for seed in args.seeds:
                set_seed(seed)
                run_dir = out_root / "checkpoints" / f"{task}" / f"bayeslora_th{th:.2f}_seed{seed}"
                cfg = TrainConfig(
                    model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                    epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                    adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                    kl_scale=args.kl_scale, prune_every=args.prune_every, prune_start=args.prune_start,
                    log_alpha_threshold=th, lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                    seeds=tuple(args.seeds), out_dir=args.out_dir, target_modules=target_modules,
                    trainable_allowlist=("classifier",),
                )

                model = create_model(num_labels=num_labels, model_name=cfg.model_name)
                out = train_bayeslora(cfg, train, val, model, tokenizer, target_modules, seed, run_dir)
                bayes_acc_logs_by_seed[seed] = out["acc_log"]
                end_ranks_per_seed.append(out["budget_total_rank"])

                # Per-seed ranks heatmap (optional; uncomment if desired)
                # plot_rank_dynamics(task, out["ranks_log"], out_root / "plots" / f"{task}_th{th:.2f}_bayeslora_seed{seed}_ranks_over_steps.png", num_layers=num_layers)

                bayes_rows.append({
                    "task": task,
                    "model": args.model,
                    "method": "BayesLoRA",
                    "log_alpha_thresh": th,
                    "final_total_rank": out["budget_total_rank"],
                    "rank_budget_used": out["budget_total_rank"],
                    "accuracy": out["best_acc"],
                    "nll": None,
                    "ece": None,
                    "params_M": out["adapter_params"] / 1e6,
                    "train_time_min": None,
                    "seed": seed,
                    "notes": f"prune_every={args.prune_every}, prune_start={int(cfg.prune_start) if cfg.prune_start is not None else int(cfg.warmup_ratio * (cfg.max_steps or cfg.epochs * math.ceil(len(train)/cfg.batch_size)))}",
                })

            # Accuracy over steps (mean±std across seeds)
            plot_accuracy_over_steps(task, bayes_acc_logs_by_seed, out_root / "plots" / f"{task}_th{th:.2f}_bayeslora_accuracy_over_steps.png", title_suffix=f"th={th:.2f}")

            # Aggregate BayesLoRA ranks across seeds -> single heatmap
            bayes_rank_paths = [
                out_root / "checkpoints" / f"{task}" / f"bayeslora_th{th:.2f}_seed{seed}" / "bayeslora_ranks_log.json"
                for seed in args.seeds
            ]
            logs_all = aggregate_rank_logs(bayes_rank_paths)
            if logs_all:
                plot_rank_dynamics(task, logs_all, out_root / "plots" / f"{task}_th{th:.2f}_bayeslora_ranks_over_steps.png", num_layers=num_layers)

            # Baseline budget from END-OF-TRAINING BayesLoRA (median over seeds)
            final_budget = int(np.median(end_ranks_per_seed)) if end_ranks_per_seed else 0
            final_budget = max(final_budget, 0)
            print(f"[{task} th={th:.2f}] Baseline rank budget (median, END-OF-TRAINING): {final_budget}", flush=True)

            # ---- AdaLoRA baseline ----
            adalora_rows = []
            adalora_acc_logs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
            if final_budget > 0:
                for seed in args.seeds:
                    set_seed(seed)
                    run_dir = out_root / "checkpoints" / f"{task}" / f"adalora_th{th:.2f}_seed{seed}"
                    device = torch.device(args.device)
                    model = create_model(num_labels=num_labels, model_name=args.model).to(device)

                    steps_per_epoch = math.ceil(len(train) / args.batch_size)
                    total_steps_default = args.epochs * steps_per_epoch
                    total_steps = args.max_steps or total_steps_default
                    total_steps = max(int(total_steps), 2)

                    # M = number of actually targeted Linear modules
                    M = sum(1 for n, m in model.named_modules()
                            if isinstance(m, nn.Linear) and any(t in n for t in target_modules))
                    if M == 0:
                        print("[warn] No targeted modules found for AdaLoRA; skipping.", flush=True)
                        continue

                    # target_r is the *average* rank per targeted module -> must be INT for PEFT alloc
                    target_r_avg = float(final_budget) / float(M)
                    target_r_int = max(0, int(round(target_r_avg)))
                    print(f"[adalora] budget={final_budget} | modules={M} | target_r(avg)={target_r_avg:.4f} | target_r(int)={target_r_int}", flush=True)

                    # Allocation schedule (80% step realloc window)
                    alloc_frac = 0.80
                    adalora_cfg = AdaLoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        init_r=args.lora_r_init,
                        target_r=target_r_int,          # <- int
                        beta1=0.85,
                        beta2=0.85,
                        tinit=0,
                        tfinal=int(total_steps * alloc_frac),
                        deltaT=100,
                        total_step=int(total_steps),
                        lora_alpha=args.lora_alpha,
                        lora_dropout=0.0,
                        target_modules=target_modules,
                        modules_to_save=["classifier"],
                    )
                    model = get_peft_model(model, adalora_cfg)
                    if hasattr(model, "set_adapter"):
                        model.set_adapter("default")

                    # Warm once so tensors/state dicts are present and telemetry works
                    with torch.no_grad():
                        k1, k2 = GLUE_TASK_TO_KEYS[task]
                        texts1 = ["warmup"] * 2
                        texts2 = (["warmup too"] * 2) if k2 is not None else None
                        tok = tokenizer(texts1, texts2, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt").to(device)
                        _ = model(**{k: tok[k] for k in ("input_ids","attention_mask","token_type_ids") if k in tok})
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                    # Unfreeze classifier robustly
                    for n, p in model.named_parameters():
                        if is_classifier_param(n):
                            p.requires_grad = True

                    # Optimizer
                    ada_params, head_params = [], []
                    for n, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if is_classifier_param(n):
                            head_params.append(p)
                        else:
                            ada_params.append(p)

                    optim_groups = []
                    if ada_params:
                        optim_groups.append({"params": ada_params, "lr": args.adapter_lr, "weight_decay": args.wd})
                    if head_params:
                        optim_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.wd})
                    optimizer = torch.optim.AdamW(optim_groups)

                    warmup_steps = int(args.warmup_ratio * total_steps)
                    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
                    eval_every = args.eval_steps or steps_per_epoch

                    # AdaLoRA rank telemetry (strict: adapter 'default')
                    adalora_rank_json = run_dir / "adalora_ranks_log.json"
                    per_mod, tot = audit_peft_ranks(model, adapter_key="default")

                    # Debug prints to verify schedule & adapter presence
                    cfg_seen = model.base_model.peft_config["default"]
                    sched = {
                        "init_r": getattr(cfg_seen, "init_r", None),
                        "target_r": getattr(cfg_seen, "target_r", None),
                        "tinit": getattr(cfg_seen, "tinit", None),
                        "tfinal": getattr(cfg_seen, "tfinal", None),
                        "deltaT": getattr(cfg_seen, "deltaT", None),
                        "total_step": getattr(cfg_seen, "total_step", None),
                    }
                    print(f"adalora schedule: {sched}", flush=True)
                    sd = get_peft_model_state_dict(model)
                    sample_keys = [(k, tuple(v.shape)) for k, v in list(sd.items())[:5]]
                    print(f"[adalora] sdict sample: {sample_keys}", flush=True)

                    log_rank_snapshot("adalora", 0, per_mod, modules_count=M, json_path=adalora_rank_json)

                    best_acc = -1.0
                    best_ckpt = run_dir / "adalora_best.pt"
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    acc_log = []
                    model.train()
                    step = 0
                    epoch = 0
                    last_tot = tot

                    while step < total_steps:
                        for batch in iter_minibatches(train, batch_size=args.batch_size, shuffle=True, seed=seed + epoch):
                            if step >= total_steps:
                                break

                            enc = preprocess_batch(batch, tokenizer, task, args.max_len)
                            enc = {k: v.to(device) for k, v in enc.items()}
                            out = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc})
                            loss = out.loss

                            lr_now = scheduler.get_last_lr()[0]
                            print_step_metrics(
                                task=task, step=step, total_steps=total_steps,
                                ce=float(loss), loss=float(loss), lr=lr_now,
                                extras=None, every=max(1, eval_every // 5)
                            )

                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()

                            # run realloc BEFORE optimizer.step(), 0-based step index
                            _adalora_update(model, step)

                            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1)
                            optimizer.step()
                            scheduler.step()

                            # light rank tick every ~100 steps
                            if ((step + 1) % 100) == 0:
                                per_mod_now, tot_now = audit_peft_ranks(model, adapter_key="default")
                                if tot_now != last_tot:
                                    print(f"[adalora@{step+1}] total_rank changed: {last_tot} -> {tot_now}", flush=True)
                                    last_tot = tot_now

                            # periodic eval + JSON logging
                            if (step % eval_every == 0) or (step == total_steps - 1):
                                per_mod, _ = audit_peft_ranks(model, adapter_key="default")
                                log_rank_snapshot("adalora", step, per_mod, modules_count=M, json_path=adalora_rank_json)

                                acc = evaluate_accuracy(model, val, tokenizer, task, args.max_len, device)
                                model.train()
                                acc_log.append({"step": step, "accuracy": acc})
                                if acc > best_acc:
                                    best_acc = acc
                                    # Save PEFT adapter weights only (robust to PEFT versions)
                                    torch.save(get_peft_model_state_dict(model), best_ckpt)

                            step += 1
                        epoch += 1

                    # Load (adapter) checkpoint leniently
                    state = torch.load(best_ckpt, map_location=device)
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        print(f"[adalora] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
                    adalora_acc_logs_by_seed[seed] = acc_log
                    adalora_rows.append({
                        "task": task, "model": args.model, "method": "AdaLoRA",
                        "log_alpha_thresh": None, "final_total_rank": final_budget, "rank_budget_used": final_budget,
                        "accuracy": best_acc, "nll": None, "ece": None, "params_M": None, "train_time_min": None,
                        "seed": seed, "notes": f"peft; budget from BayesLoRA(th={th:.2f}) end-of-training; M={M}, target_r_avg={target_r_avg:.4f}, target_r_int={target_r_int}"
                    })
            else:
                print("[warn] Final budget is 0; skipping AdaLoRA baseline.", flush=True)

            if adalora_rows:
                # Accuracy over steps (mean±std)
                plot_accuracy_over_steps(task, adalora_acc_logs_by_seed, out_root / "plots" / f"{task}_th{th:.2f}_adalora_accuracy_over_steps.png", title_suffix=f"th={th:.2f} (budget={final_budget})")
                # Aggregate AdaLoRA rank logs across seeds -> heatmap
                adalora_rank_paths = [
                    out_root / "checkpoints" / f"{task}" / f"adalora_th{th:.2f}_seed{seed}" / "adalora_ranks_log.json"
                    for seed in args.seeds
                ]
                logs_all = aggregate_rank_logs(adalora_rank_paths)
                if logs_all:
                    plot_rank_dynamics(task, logs_all, out_root / "plots" / f"{task}_th{th:.2f}_adalora_ranks_over_steps.png", num_layers=num_layers)

            # ---- Plain LoRA baseline ----
            lora_rows = []
            lora_acc_logs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
            if final_budget > 0:
                for seed in args.seeds:
                    set_seed(seed)
                    run_dir = out_root / "checkpoints" / f"{task}" / f"lora_th{th:.2f}_seed{seed}"
                    cfg_lora = TrainConfig(
                        model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                        epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                        adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                        kl_scale=args.kl_scale, prune_every=0, log_alpha_threshold=th,
                        lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                        seeds=tuple(args.seeds), out_dir=args.out_dir, target_modules=target_modules
                    )
                    model, acc, lacc = train_uniform_lora(cfg_lora, train, val, tokenizer, seed, target_modules, final_budget, run_dir, num_labels)
                    lora_acc_logs_by_seed[seed] = lacc
                    lora_rows.append({
                        "task": task, "model": args.model, "method": "LoRA",
                        "log_alpha_thresh": None, "final_total_rank": final_budget, "rank_budget_used": final_budget,
                        "accuracy": acc, "nll": None, "ece": None, "params_M": None, "train_time_min": None,
                        "seed": seed, "notes": f"uniform allocation; budget from BayesLoRA(th={th:.2f}) end-of-training"
                    })

            if lora_rows:
                # Accuracy over steps (mean±std)
                plot_accuracy_over_steps(task, lora_acc_logs_by_seed, out_root / "plots" / f"{task}_th{th:.2f}_lora_accuracy_over_steps.png", title_suffix=f"th={th:.2f} (budget={final_budget})")
                # Aggregate LoRA rank logs across seeds -> heatmap
                lora_rank_paths = [
                    out_root / "checkpoints" / f"{task}" / f"lora_th{th:.2f}_seed{seed}" / "lora_ranks_log.json"
                    for seed in args.seeds
                ]
                logs_all = aggregate_rank_logs(lora_rank_paths)
                if logs_all:
                    plot_rank_dynamics(task, logs_all, out_root / "plots" / f"{task}_th{th:.2f}_lora_ranks_over_steps.png", num_layers=num_layers)

            # Combined comparison plot
            plot_accuracy_comparison(
                task,
                bayes=bayes_acc_logs_by_seed,
                adalora=adalora_acc_logs_by_seed if adalora_rows else {},
                lora=lora_acc_logs_by_seed if lora_rows else {},
                out_path=out_root / "plots" / f"{task}_th{th:.2f}_comparison_accuracy_over_steps.png",
                title_suffix=f"th={th:.2f} (baseline budget={final_budget})"
            )

            df_th = pd.DataFrame(bayes_rows + adalora_rows + lora_rows)
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            if results_csv.exists():
                df_prev = pd.read_csv(results_csv)
                df_th = pd.concat([df_prev, df_th], ignore_index=True)
            df_th.to_csv(results_csv, index=False)
            print(f"[{task} th={th:.2f}] Done. Results appended to {results_csv}", flush=True)


if __name__ == "__main__":
    try:
        print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[env] device={torch.cuda_get_device_name(0)} capability={torch.cuda_get_device_capability(0)}", flush=True)
    except Exception as e:
        print(f"[env] torch/cuda probe failed: {e}", flush=True)
    main()