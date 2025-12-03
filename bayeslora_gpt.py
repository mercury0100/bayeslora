#!/usr/bin/env python3
# bayeslora_gpt.py
# WikiText-2 language modeling with LoRA vs BayesLoRA adapters on GPT-2/nanoGPT-ish HF models.
# - Frozen base
# - BayesLoRA: KL on from step 0 with β=1, sampling on from step 0, prune-as-you-go
# - Tokenize -> concatenate -> fixed-length blocks (nanoGPT style)

'''
python bayeslora_gpt.py \
  --method bayeslora \
  --include-mlp \
  --kl-scale 1e-6 \
  --rank 32 --lora-alpha 32 \
  --prune-every 200 --logalpha-thresh 2.0 \
  --diag-every 100 --last-k-layers 12
'''

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# ===== BayesLoRA glue (expects bayeslora/hf.py in PYTHONPATH) =====
from bayeslora.hf import (
    apply_bayeslora,
    set_bayeslora_sampling,
    bayeslora_prune, get_adapter_ranks, count_adapter_params,
    LoRAWrapper, _iter_named_linears, _matches_any, _get_parent_and_attr, _module_in_out,
    bayeslora_step,
)

# ===================== config =====================
@dataclass
class Config:
    # Model / dataset
    model_id: str = "gpt2"
    dataset: str = "wikitext2"  # "wikitext-2-raw-v1"
    block_size: int = 256

    # Method
    method: str = "bayeslora"   # "lora" or "bayeslora"

    # LoRA / BayesLoRA
    rank: int = 16
    last_k_layers: int = 6
    target_projections: str = "qvo"  # gpt2: c_attn (QKV fused) ~ {q,v}, c_proj ~ o
    include_mlp: bool = True          # include mlp.c_fc and mlp.c_proj
    lora_alpha: int = 16
    kl_scale: float = 2e-4
    logalpha_thresh: float = 3.0
    prune_every: int = 0
    min_ranks: int = 0
    prune_start_step: Optional[int] = 0  # start pruning from step 0 by default

    # Train
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 5000
    warmup_steps: int = 200
    batch_size: int = 8
    grad_accum: int = 8

    # Eval / diagnostics
    mc_eval: int = 5
    max_eval_tokens: Optional[int] = 200_000  # cap eval tokens for speed
    diag_every: int = 500
    diag_token_cap: int = 50_000  # tokens used for token-acc/ECE during diagnostics

    # System
    seed: int = 42
    dtype: str = "bfloat16"  # "bfloat16" or "float16" or "float32"


# ===================== utilities =====================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(name: str):
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    if name in ("fp16", "float16"):  return torch.float16
    return torch.float32


def expected_calibration_error(confs, trues, n_bins=15):
    # Standard ECE: confidence = max-softmax prob; truth = 1 if prediction correct else 0
    import numpy as np
    confs = np.asarray(confs, dtype=np.float32)
    trues = np.asarray(trues, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi)
        if not np.any(mask):
            continue
        acc = trues[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


# ===================== data: tokenize -> concat -> chunk =====================
class BlockDataset(Dataset):
    def __init__(self, ids: torch.Tensor, block_size: int):
        assert ids.dim() == 1
        # drop remainder so every sample is exactly block_size+1 tokens
        n_blocks = (ids.numel() - 1) // block_size
        usable = n_blocks * block_size + 1
        ids = ids[:usable]
        self.x = ids[:-1].view(n_blocks, block_size)
        self.y = ids[1:].view(n_blocks, block_size)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_wikitext2(tokenizer, block_size: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tok_concat(split):
        texts = "\n\n".join(ds[split]["text"])
        enc = tokenizer(texts, return_tensors=None, add_special_tokens=False)
        return torch.tensor(enc["input_ids"], dtype=torch.long)
    ids_train = tok_concat("train")
    ids_valid = tok_concat("validation")

    train_ds = BlockDataset(ids_train, block_size)
    valid_ds = BlockDataset(ids_valid, block_size)
    return train_ds, valid_ds


# ===================== model =====================
def _build_target_modules_gpt2(model, cfg: Config):
    try:
        n_layers = int(model.config.n_layer)
    except Exception:
        n_layers = len(model.transformer.h)
    start = max(0, n_layers - int(cfg.last_k_layers))

    want_q = "q" in cfg.target_projections
    want_v = "v" in cfg.target_projections
    want_o = ("o" in cfg.target_projections) or ("p" in cfg.target_projections)

    names = []
    for i in range(start, n_layers):
        attn = f"transformer.h.{i}.attn"
        if (want_q or want_v):
            names.append(f"{attn}.c_attn")   # fused QKV on GPT-2
        if want_o:
            names.append(f"{attn}.c_proj")   # attention out

        if cfg.include_mlp:
            mlp = f"transformer.h.{i}.mlp"
            names.append(f"{mlp}.c_fc")      # MLP expand
            names.append(f"{mlp}.c_proj")    # MLP project
    return names


def add_lora(model, cfg: Config):
    target_modules = _build_target_modules_gpt2(model, cfg)
    replaced = 0
    for name, lin in list(_iter_named_linears(model)):
        if not _matches_any(name, target_modules):
            continue
        d_out, d_in = _module_in_out(lin)
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRAWrapper(lin, d_in, d_out, r=cfg.rank, lora_alpha=cfg.lora_alpha))
        replaced += 1
    print(f"[LoRA] Wrapped {replaced} GPT-2 modules (rank={cfg.rank}, targets={cfg.target_projections}, last_k={cfg.last_k_layers}, include_mlp={cfg.include_mlp}).")

    tparams = 0
    allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A", "B"))
        p.requires_grad = req
        if req: tparams += p.numel()
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[Trainable LORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model


def add_bayeslora(model, cfg: Config):
    target_modules = _build_target_modules_gpt2(model, cfg)
    model = apply_bayeslora(
        model,
        target_modules=target_modules,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        kl_scale=cfg.kl_scale,         # per-module scale stored in wrappers
        tie_alpha_per_rank=True,
        local_reparam=True,
        freeze_base=True,
        exclude=["lm_head"],
    )
    tparams = 0
    allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A_mu","B_mu","rank_log_sigma2","A_log_sigma2","B_log_sigma2"))
        p.requires_grad = req
        if req: tparams += p.numel()
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[BayesLoRA] targets={cfg.target_projections}, last_k={cfg.last_k_layers}, include_mlp={cfg.include_mlp}")
    print(f"[Trainable BAYESLORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model


def load_model_and_tokenizer(cfg: Config):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9   # prevent length warning during raw concat tokenization

    dtype = get_dtype(cfg.dtype)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            device_map="auto",
            dtype=dtype,
        )
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id)

    # freeze base
    for p in model.parameters():
        p.requires_grad = False

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    print("model device:", next(model.parameters()).device)

    return model, tokenizer


# ===================== eval helpers =====================
@torch.no_grad()
def evaluate_perplexity(cfg: Config, model, dl: DataLoader, mc_samples: int = 1) -> Tuple[float, float]:
    """Returns (ppl_det, ppl_mc). MC ppl uses averaged predictive probs across adapter samples."""
    device = next(model.parameters()).device
    model.eval()

    total_tokens = 0
    nll_det = 0.0
    nll_mc  = 0.0

    # deterministic
    set_bayeslora_sampling(model, False)
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(input_ids=x)
        logits = out.logits  # [B, T, V]
        nll_det += F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        ).item()
        total_tokens += y.numel()
        if cfg.max_eval_tokens is not None and total_tokens >= cfg.max_eval_tokens:
            break
    ppl_det = math.exp(nll_det / max(1, total_tokens))

    # MC (if requested)
    if mc_samples > 1 and cfg.method == "bayeslora":
        processed = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                out = model(input_ids=x)
                p = out.logits.softmax(-1)  # [B, T, V]
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)
            avg_probs = probs_sum / float(mc_samples)

            nll_batch = -torch.log(
                avg_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            ).sum().item()
            nll_mc += nll_batch
            processed += y.numel()
            if cfg.max_eval_tokens is not None and processed >= cfg.max_eval_tokens:
                break
        ppl_mc = math.exp(nll_mc / max(1, processed))
    else:
        ppl_mc = ppl_det

    return ppl_det, ppl_mc


@torch.no_grad()
def evaluate_token_metrics(cfg: Config, model, dl: DataLoader, mc_samples: int = 1, token_cap: Optional[int] = None):
    """Token-level accuracy and ECE (det + MC). Confidence = max-softmax prob per token."""
    import numpy as np
    device = next(model.parameters()).device
    model.eval()

    # Deterministic pass
    set_bayeslora_sampling(model, False)
    det_confs, det_correct = [], []
    counted = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(input_ids=x)
        probs = out.logits.softmax(-1)  # [B, T, V]
        confs, preds = probs.max(dim=-1)              # [B, T]
        correct = (preds == y).view(-1)               # [B*T]
        det_confs.extend(confs.view(-1).tolist())
        det_correct.extend(correct.int().tolist())
        counted += y.numel()
        if token_cap is not None and counted >= token_cap:
            break
    tokacc_det = float(np.mean(det_correct)) if det_correct else 0.0
    ece_det = expected_calibration_error(det_confs, det_correct, n_bins=15)

    # MC pass
    if mc_samples > 1 and cfg.method == "bayeslora":
        mc_confs, mc_correct = [], []
        counted = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                out = model(input_ids=x)
                p = out.logits.softmax(-1)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)
            avg_probs = probs_sum / float(mc_samples)
            confs, preds = avg_probs.max(dim=-1)
            correct = (preds == y).view(-1)
            mc_confs.extend(confs.view(-1).tolist())
            mc_correct.extend(correct.int().tolist())
            counted += y.numel()
            if token_cap is not None and counted >= token_cap:
                break
        tokacc_mc = float(np.mean(mc_correct)) if mc_correct else 0.0
        ece_mc = expected_calibration_error(mc_confs, mc_correct, n_bins=15)
    else:
        tokacc_mc, ece_mc = tokacc_det, ece_det

    return tokacc_det, ece_det, tokacc_mc, ece_mc


# ===================== optim & training =====================
def _optimizer_and_sched(model, cfg: Config, total_steps: int):
    adapter_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adapter_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    return opt, sched


def _rebuild_opt_and_sched(model, cfg: Config, total_steps: int, current_step: int):
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)
    try:
        sched.last_epoch = max(0, current_step - 1)
    except Exception:
        pass
    return opt, sched


def train_one(cfg: Config, model, train_dl: DataLoader, valid_dl: DataLoader):
    import time
    device = next(model.parameters()).device
    model.train()

    total_steps = cfg.max_steps
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)

    batches_per_epoch = len(train_dl)
    steps_per_epoch = math.ceil(batches_per_epoch / max(1, cfg.grad_accum))
    print(f"[Train] batches/epoch={batches_per_epoch} | grad_accum={cfg.grad_accum} | "
          f"~steps/epoch={steps_per_epoch} | max_steps={total_steps} | diag_every={cfg.diag_every}")

    global_step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    last_t = time.time()
    LOG_EVERY = max(1, getattr(cfg, "diag_every", 500) // 5)  # chatty heartbeat

    # === loop for as many optimizer steps as requested ===
    while global_step < total_steps:
        for x, y in train_dl:
            if global_step >= total_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Always sample for BayesLoRA during training (no warmup)
            set_bayeslora_sampling(model, (cfg.method == "bayeslora"))

            out = model(input_ids=x)
            logits = out.logits  # [B, T, V]
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="mean"
            )

            # KL term with β=1.0 from the very first step (no freeze/warmup)
            kl_term = torch.tensor(0.0, device=device, dtype=torch.float32)
            if cfg.method == "bayeslora":
                denom = max(1, x.size(0))  # normalize per batch
                kl_term = bayeslora_step(model, beta=1.0) / float(denom)

            loss = (loss_ce / cfg.grad_accum) + (kl_term / cfg.grad_accum)
            loss.backward()
            accum += 1

            if accum >= cfg.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                # Lightweight heartbeat
                if global_step % LOG_EVERY == 0:
                    dt = time.time() - last_t
                    last_t = time.time()
                    print(f"step {global_step}/{total_steps} | loss {float(loss.item()):.4f} | "
                          f"ce {float(loss_ce.item()):.4f} | kl {float(kl_term.item()):.6f} | "
                          f"{dt:.2f}s since last")

                # Diagnostics (ppl + token acc/ECE)
                if global_step % cfg.diag_every == 0:
                    ppl_det, ppl_mc = evaluate_perplexity(
                        cfg, model, valid_dl,
                        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1)
                    )
                    tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
                        cfg, model, valid_dl,
                        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
                        token_cap=cfg.diag_token_cap
                    )
                    print(
                        f"[Diag] step {global_step}/{total_steps} | "
                        f"ppl(det) {ppl_det:.2f} | ppl(MC,{cfg.mc_eval}) {ppl_mc:.2f} | "
                        f"tok-acc(det) {tokacc_det*100:.1f}% | ECE(det) {ece_det:.3f} | "
                        f"ECE(MC,{cfg.mc_eval}) {ece_mc:.3f}"
                    )

                # prune-as-you-go (start at prune_start_step; no β gating)
                if (cfg.method == "bayeslora" and cfg.prune_every > 0 and
                    global_step >= (cfg.prune_start_step or 0) and
                    global_step % cfg.prune_every == 0):
                    before = count_adapter_params(model)
                    changed = bayeslora_prune(model, logalpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
                    after = count_adapter_params(model)
                    ranks = get_adapter_ranks(model)
                    print(f"[Prune@{global_step}] adapter params: {before} -> {after}  ranks: {ranks}")
                    if changed:
                        opt, sched = _rebuild_opt_and_sched(model, cfg, total_steps, current_step=global_step)

    set_bayeslora_sampling(model, False)
    return model


# ===================== main =====================
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--model-id", type=str, default="gpt2")
    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--method", type=str, choices=["lora","bayeslora"], required=True)

    # adapters
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--target-projections", type=str, default="qvo")
    p.add_argument("--last-k-layers", type=int, default=6)
    p.add_argument("--include-mlp", action="store_true")  # enable MLP adapters
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--kl-scale", type=float, default=2e-4)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)
    p.add_argument("--prune-every", type=int, default=0)
    p.add_argument("--min-ranks", type=int, default=0)
    p.add_argument("--prune-start-step", type=int, default=0)

    # train
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--block-size", type=int, default=256)

    # eval/diag
    p.add_argument("--mc-eval", type=int, default=5)
    p.add_argument("--max-eval-tokens", type=int, default=200000)
    p.add_argument("--diag-every", type=int, default=500)
    p.add_argument("--diag-token-cap", type=int, default=50000)

    # misc
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    cfg = Config(
        model_id=args.model_id,
        dataset=args.dataset,
        method=args.method,
        rank=args.rank, target_projections=args.target_projections, last_k_layers=args.last_k_layers,
        include_mlp=bool(args.include_mlp),
        lora_alpha=args.lora_alpha,
        kl_scale=args.kl_scale,
        logalpha_thresh=args.logalpha_thresh, prune_every=args.prune_every, min_ranks=args.min_ranks,
        prune_start_step=args.prune_start_step,
        lr=args.lr, weight_decay=args.weight_decay,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        block_size=args.block_size,
        mc_eval=args.mc_eval, max_eval_tokens=args.max_eval_tokens,
        dtype=args.dtype, seed=args.seed, diag_every=args.diag_every, diag_token_cap=args.diag_token_cap,
    )
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(cfg)
    if cfg.method == "bayeslora":
        model = add_bayeslora(model, cfg)
    else:
        model = add_lora(model, cfg)

    # datasets
    if cfg.dataset.lower() in ("wikitext2", "wikitext-2", "wikitext-2-raw-v1"):
        train_ds, valid_ds = load_wikitext2(tokenizer, cfg.block_size)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=2 if pin else 0, pin_memory=pin)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=1 if pin else 0, pin_memory=pin)

    print(f"Train blocks: {len(train_ds)} | Valid blocks: {len(valid_ds)} | Block size: {cfg.block_size}")
    if cfg.method == "bayeslora":
        print(f"[BayesLoRA] Trainable params: {count_adapter_params(model):,}")

    # train
    model = train_one(cfg, model, train_dl, valid_dl)

    # final eval
    ppl_det, ppl_mc = evaluate_perplexity(cfg, model, valid_dl, mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1))
    tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
        cfg, model, valid_dl, mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
        token_cap=cfg.max_eval_tokens
    )
    print("\n=== Results (WikiText-2) ===")
    print(f"Method: {cfg.method}")
    print(f"Perplexity (det): {ppl_det:.3f} | Perplexity (MC, {cfg.mc_eval}): {ppl_mc:.3f}")
    print(f"Token-Acc (det): {tokacc_det*100:.2f}% | ECE(det): {ece_det:.4f}")
    print(f"Token-Acc (MC):  {tokacc_mc*100:.2f}% | ECE(MC,{cfg.mc_eval}): {ece_mc:.4f}")


if __name__ == "__main__":
    main()