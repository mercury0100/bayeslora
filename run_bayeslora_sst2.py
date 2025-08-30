#!/usr/bin/env python3
"""
Quick BayesLoRA fine-tune + eval on GLUE SST-2
- Small model: distilbert-base-uncased
- Metrics: Accuracy, NLL, ECE, Brier
- Compares deterministic (BayesLoRA-Det) vs MC posterior predictive

Usage:
    python run_bayeslora_sst2.py --epochs 2 --r 8 --batch_size 32 --mc_T 8
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import local bayeslora.py (ensure it's on PYTHONPATH or in same dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from bayeslora import (
    BayesLoraConfig, apply_bayeslora,
    bayeslora_regularizers, gate_kl_weight_schedule, temperature_schedule, update_bayeslora_temperatures,
    mc_predict, set_expected_gates
)

# --------------------
# Metrics
# --------------------
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=-1) == labels).float().mean().item()

def nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return F.cross_entropy(logits, labels, reduction="mean").item()

def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    # One-vs-all for the true label
    B, C = probs.shape
    y = torch.zeros_like(probs)
    y[torch.arange(B), labels] = 1.0
    return ((probs - y) ** 2).sum(dim=-1).mean().item()

def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece_val = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece_val += (prop_in_bin * (avg_conf_in_bin - acc_in_bin).abs())
    return ece_val.item()

# --------------------
# Data
# --------------------
def prepare_sst2(tokenizer, split: str, max_len: int = 128):
    ds = load_dataset("glue", "sst2")[split]
    def tok(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=max_len)
    ds = ds.map(tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    return ds

# --------------------
# Helpers
# --------------------
def logit(x: float, eps: float = 1e-6) -> float:
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x) - math.log(1.0 - x)

def num_gate_variables(model) -> int:
    n = 0
    for m in model.modules():
        if hasattr(m, "alpha"):  # BayesLoraLinear exposes .alpha
            n += m.alpha.numel()
    return n

def init_gates_to_prior(model, prior_keep: float):
    """Set gate logits so that initial drop prob matches the prior (zero initial KL)."""
    prior_drop = 1.0 - float(prior_keep)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "alpha"):
                m.alpha.fill_(logit(prior_drop))

# --------------------
# Train/Eval loops
# --------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, total_steps, step_offset, cfg: BayesLoraConfig):
    model.train()
    step = step_offset
    ngates = num_gate_variables(model)  # size of KL; used to normalize

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # schedules FIRST (avoid buffer updates after forward)
        lambda_gate = gate_kl_weight_schedule(step, total_steps, warmup_frac=cfg.kl_warmup_frac, max_weight=cfg.kl_max_weight)
        tau = temperature_schedule(step, total_steps, start=cfg.tau_init, end=cfg.tau_final, warmup_frac=cfg.tau_warmup_frac)
        update_bayeslora_temperatures(model, tau)

        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        data_loss = out.loss

        # get raw l2 & KL, but recompute total with KL normalized per gate
        l2, kl_raw, _ = bayeslora_regularizers(
            model, lambda_l2=cfg.lambda_l2, lambda_gate=1.0, prior_keep=cfg.prior_keep
        )
        kl_per_gate = kl_raw / max(1, ngates)
        reg = cfg.lambda_l2 * l2 + lambda_gate * kl_per_gate
        loss = data_loss + reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i % 100 == 0:
            print(
                f"step {step} | loss={loss.item():.4f} | data={data_loss.item():.4f} "
                f"| l2={l2.item():.4f} | kl_raw={kl_raw.item():.4f} | kl_per_gate={kl_per_gate.item():.6f} "
                f"| tau={tau:.3f} | lambda_gate={lambda_gate:.3f} | gates={ngates}"
            )

        step += 1
    return step

@torch.no_grad()
def evaluate_det(model, loader, device):
    model.eval()
    set_expected_gates(model, True)   # deterministic BayesLoRA-Det
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        all_logits.append(logits)
        all_labels.append(batch["label"])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = logits.softmax(dim=-1)
    return {
        "acc": accuracy(logits, labels),
        "nll": nll_loss(logits, labels),
        "ece": ece(probs, labels),
        "brier": brier_score(probs, labels),
    }

@torch.no_grad()
def evaluate_mc(model, loader, device, T: int = 8, antithetic: bool = True):
    model.eval()
    set_expected_gates(model, False)  # allow sampling
    all_mean, all_var, all_labels = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        def forward_once():
            return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        mean, var = mc_predict(model, forward_once, T=T, antithetic=antithetic)
        all_mean.append(mean)
        all_var.append(var)
        all_labels.append(batch["label"])
    mean = torch.cat(all_mean, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = mean.softmax(dim=-1)
    return {
        "acc": accuracy(mean, labels),
        "nll": nll_loss(mean, labels),
        "ece": ece(probs, labels),
        "brier": brier_score(probs, labels),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--prior_keep", type=float, default=0.7)
    parser.add_argument("--tau_init", type=float, default=2.0)
    parser.add_argument("--tau_final", type=float, default=0.5)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.3)
    parser.add_argument("--tau_warmup_frac", type=float, default=0.3)
    parser.add_argument("--kl_max_weight", type=float, default=0.05)  # gentler cap
    parser.add_argument("--lambda_l2", type=float, default=1e-4)
    parser.add_argument("--mc_T", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = prepare_sst2(tokenizer, "train", max_len=args.max_len)
    val_ds = prepare_sst2(tokenizer, "validation", max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Apply BayesLoRA BEFORE moving to device so new params follow
    cfg = BayesLoraConfig(
        r=args.r, lora_alpha=args.lora_alpha,
        prior_keep=args.prior_keep,
        tau_init=args.tau_init, tau_final=args.tau_final,
        kl_warmup_frac=args.kl_warmup_frac, tau_warmup_frac=args.tau_warmup_frac,
        kl_max_weight=args.kl_max_weight, lambda_l2=args.lambda_l2,
        # DistilBERT module names
        target_modules=("q_lin", "k_lin", "v_lin", "out_lin", "pre_classifier"),
        # per_module_shared_mask=False,  # set True for 1 gate/module if you need stronger regularization
        # freeze_base=True  # default
    )
    apply_bayeslora(model, cfg, verbose=True)

    # Initialize gate logits to the prior (zero initial KL)
    init_gates_to_prior(model, prior_keep=cfg.prior_keep)

    # NOW move the entire model (including adapters) to device
    model.to(device)

    # Optimizer & scheduler
    num_training_steps = args.epochs * len(train_loader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Train
    step = 0
    for epoch in range(args.epochs):
        step = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                               total_steps=num_training_steps, step_offset=step, cfg=cfg)

        # Quick eval each epoch
        det = evaluate_det(model, val_loader, device)
        mc = evaluate_mc(model, val_loader, device, T=args.mc_T, antithetic=True)
        print(f"\n[Epoch {epoch+1}] Validation metrics")
        print("  Deterministic:", det)
        print("  MC (T={}):   ".format(args.mc_T), mc)

    # Final report
    det = evaluate_det(model, val_loader, device)
    mc = evaluate_mc(model, val_loader, device, T=args.mc_T, antithetic=True)
    print("\n=== Final Validation ===")
    print("Deterministic:", det)
    print("MC (T={}):   ".format(args.mc_T), mc)

if __name__ == "__main__":
    main()
