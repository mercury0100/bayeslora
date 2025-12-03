#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLUE helpers: data, metrics, plotting, and unified trainers for:
- BayesLoRA
- AdaLoRA (PEFT)
- LoRA (fixed, uniform allocation)

All trainers expose the SAME interface:

def train_*(cfg: TrainConfig,
            train, val,
            model, tokenizer,
            target_modules: List[str],
            seed: int,
            run_dir: Path) -> Dict[str, Any]

Returns a dict:
{
  "model": best_model_on_device,         # torch.nn.Module (weights loaded to cfg.device)
  "best_acc": float,                     # best validation accuracy seen
  "acc_log": List[{"step": int, "accuracy": float, "ema": float?}],
  "ranks_log": List[{"step": int, "total_rank": int, "per_module": {str:int}}],
  "budget_ranks": Dict[str,int],         # last snapshot (end-of-training ranks)
  "budget_total_rank": int,              # sum of end-of-training ranks (used as budget)
  "adapter_params": int,                 # number of trainable adapter parameters
  "runtime_min": float                   # wall-clock minutes for training routine
}

Notes
- BayesLoRA uses `bayeslora.hf` wrappers.
- AdaLoRA is via PEFT (AdaLoraConfig).
- LoRA fixed baseline uses a tiny internal wrapper (compatible with your previous LoRAWrapper).
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_JIT", "0")

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

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

# ---- Optional deps that may not exist at import-time in some envs ----
# BayesLoRA & LoRA wrappers (your hf.py)
try:
    from bayeslora.hf import (
        apply_bayeslora,
        bayeslora_step,
        bayeslora_prune,
        get_adapter_ranks,
        count_adapter_params,
        set_bayeslora_sampling,
    )
except Exception as _e:
    # Provide soft fallbacks so the module can import; will raise if actually used
    apply_bayeslora = bayeslora_step = bayeslora_prune = None
    get_adapter_ranks = count_adapter_params = set_bayeslora_sampling = None
    LoRAWrapper = None

# PEFT / AdaLoRA
try:
    from peft import get_peft_model, AdaLoraConfig, TaskType
    from peft.utils import get_peft_model_state_dict
except Exception:
    get_peft_model = AdaLoraConfig = TaskType = None
    get_peft_model_state_dict = None


# -------------------------
# Seed / Constants
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


# -------------------------
# Data / Preprocessing
# -------------------------
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


def iter_minibatches(dataset, batch_size: int, shuffle: bool, seed: Optional[int]) -> Iterable[List[dict]]:
    idxs = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    for i in range(0, len(dataset), batch_size):
        yield [dataset[j] for j in idxs[i:i+batch_size]]


# -------------------------
# Tiny utils
# -------------------------
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
    # neutral: keep original order rather than sorted alphabetical
    for i in range(extra):
        out[target_names[i % len(target_names)]] += 1
    return out


# -------------------------
# AdaLoRA telemetry (STATE-DICT ONLY; PEFT 0.17.x SVDLinear-safe)
# -------------------------
def _parse_sdict_for_ranks(state_dict_like):
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
            IND_nz[(mod, ad)] = int((v != 0).sum().item()); continue

        mod, ad = split_key(k, "lora_E")
        if mod is not None:
            if v.ndim == 2:
                E_rows_nz[(mod, ad)] = int((v.abs().max(dim=1).values > 0).sum().item())
            else:
                E_rows_nz[(mod, ad)] = int((v != 0).sum().item())
            continue

        mod, ad = split_key(k, "lora_A")
        if mod is not None:
            A_rows[(mod, ad)] = int(v.shape[0]) if v.ndim == 2 else 0; continue

        mod, ad = split_key(k, "lora_B")
        if mod is not None:
            B_dim[(mod, ad)] = int(min(v.shape)) if v.ndim == 2 else 0; continue

    per, total = {}, 0
    keys = set().union(A_rows.keys(), B_dim.keys(), E_rows_nz.keys(), IND_nz.keys())
    for mk in keys:
        r_ind = IND_nz.get(mk, 0)
        r_e   = E_rows_nz.get(mk, 0)
        rA    = A_rows.get(mk, 0)
        rB    = B_dim.get(mk, 0)
        r_shape = rA if (rB == 0) else (rB if (rA == 0) else min(rA, rB))
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
    # Prefer PEFT model state dict, fallback to full model
    sdict = None
    try:
        sdict = get_peft_model_state_dict(model) if get_peft_model_state_dict else None
    except Exception:
        sdict = None

    if sdict and any((".lora_" in k) for k in sdict.keys()):
        return _parse_sdict_for_ranks(sdict)

    full_sd = None
    try:
        full_sd = model.state_dict()
    except Exception:
        full_sd = None

    if full_sd and any((".lora_" in k) for k in full_sd.keys()):
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


def log_rank_snapshot(method: str, step: int, per_mod: Dict[str, int], modules_count: int, json_path: Optional[Path] = None):
    """If json_path is provided, also append a minimal snapshot file; else just print."""
    tot = int(sum(per_mod.values()))
    avg = tot / max(1, modules_count)
    print(f"[{method}@{step}] total_rank={tot} (avg={avg:.4f})", flush=True)
    if json_path is not None:
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
def plot_rank_dynamics(task: str,
                       logs: List[Dict[str, Any]],
                       out_path: Path,
                       num_layers: Optional[int] = None):
    """Two aligned heatmaps."""
    if not logs:
        return

    steps = sorted({int(x["step"]) for x in logs})
    if not steps:
        return
    if steps[0] != 0:
        steps = [0] + steps
    steps = np.asarray(steps, dtype=float)
    smax = steps[-1]

    def _nice_step(x):
        if x <= 0: return 1.0
        raw = x / 5.0
        mag = 10 ** np.floor(np.log10(raw))
        frac = raw / mag
        return (1.0 if frac < 1.5 else 2.0 if frac < 3.5 else 5.0) * mag

    tick_step  = _nice_step(max(smax, 1.0))
    x_max_nice = tick_step * np.ceil((smax + 1e-9) / tick_step)
    step_edges = np.concatenate([steps, [x_max_nice]])
    step_to_col = {int(s): i for i, s in enumerate(steps)}
    T = len(steps)

    if num_layers is None:
        seen = set()
        for x in logs:
            for n in x["per_module"].keys():
                parts = n.split(".")
                for i, p in enumerate(parts):
                    if p == "layer" and i + 1 < len(parts):
                        try: seen.add(int(parts[i+1]))
                        except: pass
        num_layers = (max(seen) + 1) if seen else 1

    def extract_layer(n: str):
        parts = n.split(".")
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts):
                try:
                    L = int(parts[i+1]);  return L if 0 <= L < num_layers else None
                except:
                    return None
        return None

    MODULE_LABELS = ["Q", "K", "V", "O", "MLP_in", "MLP_out"]
    SUBSTR = {
        "Q": "attention.self.query_proj",
        "K": "attention.self.key_proj",
        "V": "attention.self.value_proj",
        "O": "attention.output.dense",
        "MLP_in": "intermediate.dense",
        "MLP_out": "output.dense",
    }
    lab2row = {lab: i for i, lab in enumerate(MODULE_LABELS)}
    def which_module(n: str):
        for lab, sub in SUBSTR.items():
            if sub in n: return lab
        return None

    mat_layers  = np.zeros((num_layers, T), dtype=float)
    mat_modules = np.zeros((len(MODULE_LABELS), T), dtype=float)
    for entry in logs:
        s = int(entry["step"])
        col = step_to_col.get(s)
        if col is None:
            continue
        for name, r in entry["per_module"].items():
            v = float(r)
            L = extract_layer(name)
            if L is not None:
                mat_layers[L, col] += v
            m = which_module(name)
            if m is not None:
                mat_modules[lab2row[m], col] += v

    vmin = float(min(mat_layers.min(), mat_modules.min()))
    vmax = float(max(mat_layers.max(), mat_modules.max()))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13, 6.5))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1.0, 0.045],
                  height_ratios=[2.0, 2.0],
                  hspace=0.25, wspace=0.10)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[:, 1])

    layer_edges  = np.arange(-0.5, num_layers + 0.5, 1.0)
    module_edges = np.arange(-0.5, len(MODULE_LABELS) + 0.5, 1.0)

    X1, Y1 = np.meshgrid(step_edges, layer_edges)
    im1 = ax1.pcolormesh(X1, Y1, mat_layers, cmap="viridis", norm=norm, shading="flat")
    ax1.set_title(f"{task.upper()} – Sum of adapter ranks per layer")
    ax1.set_ylabel("Encoder layer")
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels([str(i) for i in range(num_layers)])

    X2, Y2 = np.meshgrid(step_edges, module_edges)
    im2 = ax2.pcolormesh(X2, Y2, mat_modules, cmap="viridis", norm=norm, shading="flat")
    ax2.set_title(f"{task.upper()} – Sum of adapter ranks per module")
    ax2.set_ylabel("Module")
    ax2.set_yticks(range(len(MODULE_LABELS)))
    ax2.set_yticklabels(MODULE_LABELS)
    ax2.set_xlabel("Step")

    cb = fig.colorbar(im2, cax=cax); cb.set_label("Sum of ranks")

    for ax in (ax1, ax2):
        ax.set_xlim(0.0, x_max_nice)
    xticks = np.arange(0.0, x_max_nice + 0.5 * tick_step, tick_step)
    ax2.set_xticks(xticks)
    ax1.label_outer()

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        step_to_val = {x["step"]: x.get("accuracy") for x in logs}
        curves.append([step_to_val.get(s, np.nan) for s in all_steps])
    M = np.array(curves, dtype=float)
    mean = np.nanmean(M, axis=0)
    std  = np.nanstd(M, axis=0)

    fig = plt.figure(figsize=(10.5, 4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(all_steps, mean, linewidth=2, label="mean acc")
    ax.fill_between(all_steps, mean-std, mean+std, alpha=0.2, label="±1 std")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ttl = f"{task.upper()} – Accuracy over steps (mean±std across seeds)"
    if title_suffix: ttl += f" – {title_suffix}"
    ax.set_title(ttl); ax.grid(True, linestyle="--", alpha=0.3); ax.legend()

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
            m = {x["step"]: x.get("accuracy") for x in logs}
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

    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ttl = f"{task.upper()} – Accuracy comparison (mean±std)"
    if title_suffix: ttl += f" – {title_suffix}"
    ax.set_title(ttl); ax.grid(True, linestyle="--", alpha=0.3); ax.legend()

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
        try:
            if set_bayeslora_sampling:
                set_bayeslora_sampling(model, False)
        except Exception:
            pass
        logits = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc}).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        acc.add_batch(predictions=preds, references=enc["labels"].cpu().numpy())
    return float(acc.compute()["accuracy"])


@torch.no_grad()
def evaluate_metrics(model,
                     dataset,
                     tokenizer,
                     task,
                     max_len,
                     device,
                     batch_size: int = 128,
                     n_bins: int = 15) -> dict:
    """Returns: {"accuracy": float, "nll": float, "ece": float}"""
    model.eval()
    correct = 0
    total = 0
    nll_sum = 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_correct = np.zeros(n_bins, dtype=np.int64)
    bin_count   = np.zeros(n_bins, dtype=np.int64)
    bin_conf    = np.zeros(n_bins, dtype=np.float64)

    for batch in iter_minibatches(dataset, batch_size=batch_size, shuffle=False, seed=1234):
        enc = preprocess_batch(batch, tokenizer, task, max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**{k: enc[k] for k in ("input_ids","attention_mask","token_type_ids","labels") if k in enc}).logits
        labels = enc["labels"]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        pred = probs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.numel()
        nll_sum += (-log_probs.gather(1, labels.view(-1,1)).squeeze(1)).sum().item()

        conf, argmax = probs.max(dim=-1)
        eq = (argmax == labels).to(torch.int32).cpu().numpy()
        c  = conf.detach().cpu().numpy()
        inds = np.minimum(np.searchsorted(bin_edges, c, side="right") - 1, n_bins - 1)
        for i_bin in range(n_bins):
            mask = (inds == i_bin)
            if not np.any(mask): continue
            bin_count[i_bin]   += int(mask.sum())
            bin_correct[i_bin] += int(eq[mask].sum())
            bin_conf[i_bin]    += float(c[mask].sum())

    acc = (correct / max(1, total))
    nll = (nll_sum / max(1, total))

    ece = 0.0
    for i in range(n_bins):
        if bin_count[i] == 0: continue
        avg_conf = bin_conf[i] / bin_count[i]
        avg_acc  = bin_correct[i] / bin_count[i]
        ece += (bin_count[i] / max(1, total)) * abs(avg_acc - avg_conf)

    return {"accuracy": float(acc), "nll": float(nll), "ece": float(ece)}


def count_peft_adapter_params(model) -> int:
    """For LoRA/AdaLoRA via PEFT: count ONLY adapter tensors."""
    try:
        from peft.utils import get_peft_model_state_dict as _get
        sdict = _get(model)
        return int(sum(v.numel() for v in sdict.values()))
    except Exception:
        total = 0
        for n, p in model.named_parameters():
            if (".lora_" in n) or ("lora_A" in n) or ("lora_B" in n):
                total += p.numel()
        return int(total)


def wallclock_minutes(t_start: float) -> float:
    return float((time.time() - t_start) / 60.0)


# ---------- No KL floor (you control kl_scale) ----------
def set_kl_scale(model: nn.Module, new_scale: float):
    for m in model.modules():
        if hasattr(m, "_kl_scale"):
            m._kl_scale = float(new_scale)

def scaled_kl(model: nn.Module) -> torch.Tensor:
    if not bayeslora_step:
        raise RuntimeError("bayeslora_step is unavailable. Is bayeslora.hf installed?")
    return bayeslora_step(model, 1.0)
# -------------------------------------------------------


def build_param_groups(model: nn.Module, adapter_lr: float, head_lr: float, wd: float):
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
            logvars.append(p)
        elif is_adapter_mean(n):
            adapter_means.append(p)
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


# =======================================================
# Unified Trainers
# =======================================================

def _warm_kernel(model, tokenizer, task, max_len, device):
    k1, k2 = GLUE_TASK_TO_KEYS[task]
    texts1 = ["hello world"] * 2
    texts2 = ["hello there"] * 2 if k2 is not None else None
    tok = tokenizer(texts1, texts2, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        _ = model(**{k: tok[k] for k in ("input_ids","attention_mask","token_type_ids") if k in tok}).logits
    if device.type == "cuda":
        torch.cuda.synchronize()


def _finalize_result(start_time: float,
                     model: nn.Module,
                     best_model_snapshot: Optional[nn.Module],
                     device: torch.device,
                     ranks_log: List[Dict[str, Any]],
                     acc_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    if best_model_snapshot is not None:
        model = best_model_snapshot.to(device)
    runtime_min = wallclock_minutes(start_time)
    # Rank budget = last snapshot (or zeros)
    if ranks_log:
        end_ranks = ranks_log[-1].get("per_module", {})
        end_total_rank = int(sum(end_ranks.values()))
    else:
        end_ranks, end_total_rank = {}, 0
    try:
        adapter_params = count_adapter_params(model)
    except Exception:
        adapter_params = count_peft_adapter_params(model)
    best_acc = max((x.get("accuracy", float("-inf")) for x in acc_log), default=-1.0)
    return {
        "model": model,
        "best_acc": float(best_acc),
        "acc_log": acc_log,
        "ranks_log": ranks_log,
        "budget_ranks": end_ranks,
        "budget_total_rank": int(end_total_rank),
        "adapter_params": int(adapter_params),
        "runtime_min": float(runtime_min),
    }


def train_bayeslora(cfg: TrainConfig,
                    train, val,
                    model, tokenizer,
                    target_modules: List[str],
                    seed: int,
                    run_dir: Path) -> Dict[str, Any]:
    if not apply_bayeslora or not get_adapter_ranks:
        raise RuntimeError("BayesLoRA wrappers are unavailable. Ensure bayeslora.hf is installed.")

    t0 = time.time()
    device = torch.device(cfg.device)
    model.to(device)

    # Apply BayesLoRA
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

    _warm_kernel(model, tokenizer, cfg.task, cfg.max_len, device)

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps_default = cfg.epochs * steps_per_epoch
    total_steps = cfg.max_steps or total_steps_default
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    prune_start = cfg.prune_start if (cfg.prune_start is not None) else warmup_steps
    eval_every = cfg.eval_steps or steps_per_epoch

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
    best_model_snapshot = None
    run_dir.mkdir(parents=True, exist_ok=True)

    r0 = module_total_rank(get_adapter_ranks(model))
    print(f"[bayeslora@init] total_rank={r0}", flush=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sanity] kl_scale={cfg.kl_scale:.3e} | trainable_params={n_trainable/1e6:.3f}M "
          f"| lr(adapter)={cfg.adapter_lr:.3e} | lr(head)={cfg.head_lr:.3e} "
          f"| warmup_steps={warmup_steps} | prune_start={prune_start} | steps/epoch={steps_per_epoch}", flush=True)

    model.train()
    step = 0
    epoch = 0
    set_seed(seed)

    while step < total_steps:
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

            if (cfg.prune_every > 0) and (step >= prune_start) and (step % cfg.prune_every == 0):
                pre_total = module_total_rank(get_adapter_ranks(model))
                changed = bayeslora_prune(model, logalpha_thresh=cfg.log_alpha_threshold, min_ranks=0)
                post_ranks = get_adapter_ranks(model)
                post_total = module_total_rank(post_ranks)
                print(f"[prune@{step}] th={cfg.log_alpha_threshold:.2f} | total_rank {pre_total} -> {post_total} | changed={bool(changed)}", flush=True)
                optimizer, scheduler = make_optim_and_sched(step_already=step)

            if any(p.requires_grad for p in model.parameters()):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.clip_grad)
                optimizer.step()
                scheduler.step()

            if (step % eval_every == 0) or (step == total_steps - 1):
                ranks = get_adapter_ranks(model)
                total_rank = module_total_rank(ranks)
                ranks_log.append({"step": step, "total_rank": total_rank, "per_module": ranks})
                acc = evaluate_accuracy(model, val, tokenizer, cfg.task, cfg.max_len, device)
                model.train()
                ema_acc = ema(ema_acc, acc, beta=0.9)
                acc_log.append({"step": step, "accuracy": float(acc), "ema": float(ema_acc) if ema_acc is not None else None})
                if acc > best_acc:
                    best_acc = acc
                    best_model_snapshot = copy.deepcopy(model).to("cpu")

            step += 1
        epoch += 1

    # Persist logs for external plotting
    save_json(ranks_log, run_dir / "bayeslora_ranks_log.json")
    save_json(acc_log,   run_dir / "bayeslora_accuracy_log.json")

    return _finalize_result(
        start_time=t0,
        model=model,
        best_model_snapshot=best_model_snapshot,
        device=device,
        ranks_log=ranks_log,
        acc_log=acc_log,
    )

# -------------------------
# AdaLoRA helpers & trainer
# -------------------------
def _adalora_update(model, global_step_0based: int, hard_budget: Optional[int] = None, enforce_non_increasing: bool = False) -> bool:
    """
    Call AdaLoRA's reallocation regardless of wrapper depth.
    """
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]

    for tgt in candidates:
        if tgt is None or not hasattr(tgt, "update_and_allocate"):
            continue

        alloc = getattr(tgt, "rankallocator", None)

        if alloc is not None:
            scores = getattr(alloc, "scores", None)
            if scores and all((s is None or (hasattr(s, "numel") and s.numel() == 0)) for s in scores.values()):
                print(f"[adalora@{global_step_0based}] skip empty allocator for {tgt.__class__.__name__}")
                continue

            if hard_budget is not None and hasattr(alloc, "rank_budget"):
                try:
                    hb = int(hard_budget)
                except Exception:
                    hb = 1
                hb = max(1, hb)
                if enforce_non_increasing:
                    try:
                        current = int(getattr(alloc, "rank_budget", hb))
                    except Exception:
                        current = hb
                    hb = min(current, hb)
                alloc.rank_budget = hb
            elif hasattr(alloc, "rank_budget"):
                try:
                    alloc.rank_budget = max(1, int(getattr(alloc, "rank_budget", 1)))
                except Exception:
                    alloc.rank_budget = 1

        try:
            tgt.update_and_allocate(global_step_0based)
        except Exception as e:
            print(f"[adalora@{global_step_0based}] update_and_allocate failed on {tgt.__class__.__name__}: {e}")
            continue

        return True

    return False


# -------------------------
# AdaLoRA (PEFT) — with telemetry
# -------------------------
def train_adalora(
    cfg: TrainConfig,
    train,
    val,
    model,
    tokenizer,
    target_modules: List[str],
    seed: int,
    run_dir: Path,
    budget_total_rank: int,
):
    from peft import get_peft_model, AdaLoraConfig, TaskType
    from peft.utils import get_peft_model_state_dict

    t0 = time.time()
    device = torch.device(cfg.device)
    model = model.to(device)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps_default = cfg.epochs * steps_per_epoch
    total_steps = max(2, int(cfg.max_steps or total_steps_default))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    eval_every = steps_per_epoch if (cfg.eval_steps is None) else int(cfg.eval_steps)

    # Count targeted Linear modules and derive target_r conservatively (floor)
    M = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(t in n for t in target_modules)
    )
    if M == 0:
        raise RuntimeError("AdaLoRA: no targeted Linear modules found.")
    target_r_int = max(1, round(budget_total_rank / M))
    tfinal = int(total_steps * 0.80)

    ada_cfg = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=cfg.lora_r_init,
        target_r=target_r_int,         # per-module target rank (floor)
        beta1=0.85, beta2=0.85,
        tinit=warmup_steps,
        tfinal=tfinal,
        deltaT=200,
        total_step=int(total_steps),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, ada_cfg)
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")

    # Warm kernels (keeps first forward deterministic and allocators happy)
    with torch.no_grad():
        k1, k2 = GLUE_TASK_TO_KEYS[cfg.task]
        txt1 = ["warmup"] * 2
        txt2 = (["warmup too"] * 2) if k2 is not None else None
        tok = tokenizer(txt1, txt2, truncation=True, padding=True,
                        max_length=cfg.max_len if hasattr(cfg, "max_len") else cfg.max_len,
                        return_tensors="pt").to(device)
        _ = model(**{k: tok[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tok})
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Train heads + adapters (enable heads explicitly; PEFT already enables adapters)
    for n, p in model.named_parameters():
        if is_classifier_param(n):
            p.requires_grad = True

    ada_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if is_classifier_param(n) else ada_params).append(p)

    optim_groups = []
    if ada_params:
        optim_groups.append({"params": ada_params, "lr": cfg.adapter_lr, "weight_decay": cfg.wd})  # no WD on adapters
    if head_params:
        optim_groups.append({"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.wd})
    optimizer = torch.optim.AdamW(optim_groups)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- step-0 snapshots + telemetry ----
    per_mod0, tot0 = audit_peft_ranks(model, adapter_key="default")
    ranks_log = [{"step": 0, "total_rank": int(tot0), "per_module": per_mod0}]
    log_rank_snapshot("adalora", 0, per_mod0, modules_count=M, json_path=run_dir / "adalora_ranks_log.json")

    acc0 = evaluate_accuracy(model, val, tokenizer, cfg.task, cfg.max_len, device)
    acc_log = [{"step": 0, "accuracy": float(acc0)}]
    best_model_snapshot = copy.deepcopy(model).to("cpu")

    print(
        f"[sanity:adalora] budget_total_rank={int(budget_total_rank)} | modules={M} | "
        f"target_r(int)={target_r_int} | "
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M | "
        f"tinit={warmup_steps} tfinal={tfinal} deltaT=200 total_step={total_steps}",
        flush=True,
    )

    # ---- train ----
    set_seed(seed)
    model.train()
    step = 0
    epoch = 0
    last_tot = int(tot0)

    while step < total_steps:
        for batch in iter_minibatches(train, batch_size=cfg.batch_size, shuffle=True, seed=seed + epoch):
            if step >= total_steps:
                break

            enc = preprocess_batch(batch, tokenizer, cfg.task, cfg.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**{k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids", "labels") if k in enc})
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # reallocate/enforce global budget every step
            _adalora_update(model, step, hard_budget=int(budget_total_rank), enforce_non_increasing=True)
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.clip_grad)
            optimizer.step()
            scheduler.step()

            # Telemetry
            lr_now = scheduler.get_last_lr()[0]
            print_step_metrics(
                task=cfg.task if hasattr(cfg, "task") else cfg.task,
                step=step, total_steps=total_steps,
                ce=float(loss), loss=float(loss), lr=lr_now,
                extras=None, every=max(1, eval_every // 5)
            )

            # Optional rank-change notice
            if ((step + 1) % 100) == 0:
                _, tot_probe = audit_peft_ranks(model, adapter_key="default")
                if int(tot_probe) != last_tot:
                    print(f"[adalora@{step+1}] total_rank changed: {last_tot} -> {int(tot_probe)}", flush=True)
                    last_tot = int(tot_probe)

            # strict cadence logging/eval
            if (step % eval_every == 0) or (step == total_steps - 1):
                per_mod_now, tot_now = audit_peft_ranks(model, adapter_key="default")
                ranks_log.append({"step": int(step), "total_rank": int(tot_now), "per_module": per_mod_now})
                log_rank_snapshot("adalora", step, per_mod_now, modules_count=M, json_path=None)

                acc = evaluate_accuracy(model, val, tokenizer, cfg.task, cfg.max_len, device)
                model.train()
                acc_log.append({"step": int(step), "accuracy": float(acc)})

                # Keep snapshot if it's the best so far (or keep your tfinal rule if you prefer)
                # Using anytime-best keeps parity with your other trainers' finalize
                if acc >= max(x["accuracy"] for x in acc_log[:-1]):  # best-so-far
                    best_model_snapshot = copy.deepcopy(model).to("cpu")

                last_tot = int(tot_now)
            step += 1
        epoch += 1

    # Final hard clamp to ensure the audit matches the budget
    _adalora_update(model, step, hard_budget=int(budget_total_rank), enforce_non_increasing=True)
    per_mod_final, tot_final = audit_peft_ranks(model, adapter_key="default")
    if int(tot_final) > int(budget_total_rank):
        print(f"[warn] AdaLoRA final rank {int(tot_final)} > budget {int(budget_total_rank)}; clamping again.", flush=True)
        _adalora_update(model, step, hard_budget=int(budget_total_rank), enforce_non_increasing=True)
        per_mod_final, tot_final = audit_peft_ranks(model, adapter_key="default")

    # persist logs
    save_json(ranks_log, run_dir / "adalora_ranks_log.json")
    save_json(acc_log,   run_dir / "adalora_acc_log.json")

    # finalize (returns model snapshot w/ best acc, ranks/acc logs, params, runtime)
    return _finalize_result(t0, model, best_model_snapshot, device, ranks_log, acc_log)

# -------------------------
# Plot aggregation utilities
# -------------------------
def mean_std_on_grid(logs_by_seed: Dict[int, List[Dict[str, Any]]], value_key: str, grid: np.ndarray, min_seeds: int = 2):
    """NaN-aware aggregation without forward-fill. Returns (mean, std, counts)."""
    if not logs_by_seed:
        n = len(grid); z = np.full(n, np.nan, dtype=float)
        return z, z, np.zeros(n, dtype=int)

    rows = []
    for seed, logs in logs_by_seed.items():
        d = {int(e["step"]): float(e[value_key]) for e in logs if value_key in e and e.get(value_key) is not None}
        rows.append(np.array([d.get(int(s), np.nan) for s in grid], dtype=float))
    M = np.vstack(rows)
    counts = np.sum(~np.isnan(M), axis=0)
    mean = np.where(counts >= min_seeds, np.nanmean(M, axis=0), np.nan)
    std  = np.where(counts >= min_seeds, np.nanstd (M, axis=0), np.nan)
    return mean, std, counts


def common_steps(logs_by_seed: Dict[int, List[Dict[str, Any]]]) -> np.ndarray:
    """Intersection of step sets across seeds (sorted)."""
    step_sets = [set(int(e["step"]) for e in logs) for logs in logs_by_seed.values() if logs]
    return np.array(sorted(set.intersection(*step_sets))) if step_sets else np.array([], dtype=int)