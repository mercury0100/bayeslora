#!/usr/bin/env python3
# bayeslora_llama.py
# Straightforward BayesLoRA SFT for Llama 3.2 1B on the cleaned Alpaca dataset.
# - No beta schedule / no KL freeze / no sampling warmup / no beta prune threshold
# - Keeps prune-as-you-go + linear LR warmup schedule
# - Masks prompts (loss only on the response)
# - Optional inclusion of MLP adapters (gate/up/down proj)
#
# Example:
'''
python bayeslora_llama.py \
  --method bayeslora \
  --dataset vicgalle/alpaca-gpt4 \
  --include-mlp \
  --rank 16 --lora-alpha 16 \
  --kl-scale 1e-7 \
  --max-seq-len 1024 \
  --prune-every 200 --logalpha-thresh 3.0
'''
'''
python bayeslora_llama.py \
  --method lora \
  --dataset vicgalle/alpaca-gpt4
  --include-mlp \
  --rank 16 --lora-alpha 16 \
  --max-seq-len 1024 \
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    # Model / data
    model_id: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "yahma/alpaca-cleaned"
    val_ratio: float = 0.01       # make a small validation split from train
    max_seq_len: int = 1024

    # Method
    method: str = "bayeslora"     # "lora" or "bayeslora"

    # Adapters
    rank: int = 16
    last_k_layers: int = 9999     # adapt all layers by default (set smaller to limit)
    target_projections: str = "qvo"  # attention: any of 'q','k','v','o' (we default to qvo)
    include_mlp: bool = True
    lora_alpha: int = 16
    kl_scale: float = 2e-4
    logalpha_thresh: float = 3.0
    prune_every: int = 0
    min_ranks: int = 1
    prune_start_step: Optional[int] = 0  # prune from step 0

    # Train
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 5000
    warmup_steps: int = 200
    batch_size: int = 8
    grad_accum: int = 8

    # Eval / diagnostics
    diag_every: int = 500
    mc_eval: int = 5
    max_eval_tokens: Optional[int] = 200_000

    # System
    seed: int = 42
    dtype: str = "bfloat16"        # "bfloat16" | "float16" | "float32"


# ===================== utils =====================
def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_dtype(name: str):
    name = (name or "").lower()
    if name in ("bf16","bfloat16"): return torch.bfloat16
    if name in ("fp16","float16"):  return torch.float16
    return torch.float32

def expected_calibration_error(confs, trues, n_bins=15):
    import numpy as np
    confs = np.asarray(confs, dtype=np.float32)
    trues = np.asarray(trues, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi)
        if not np.any(mask): continue
        acc  = trues[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

# ===================== data: Alpaca SFT =====================
ALPACA_PROMPT_W_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
ALPACA_PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

class AlpacaSFTDataset(Dataset):
    def __init__(self, tokenizer, records, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = int(max_seq_len)
        self.examples: List[Dict[str, torch.Tensor]] = []

        eos = self.tok.eos_token or ""
        for ex in records:
            instr = (ex.get("instruction") or "").strip()
            inp   = (ex.get("input") or "").strip()
            out   = (ex.get("output") or "").strip()

            if inp:
                prompt = ALPACA_PROMPT_W_INPUT.format(instruction=instr, input=inp, output="")
            else:
                prompt = ALPACA_PROMPT_NO_INPUT.format(instruction=instr, output="")

            # We want labels only for the response tokens
            resp = out + ("" if out.endswith(eos) else eos)

            # Tokenize separately to know the prompt length
            prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
            resp_ids   = self.tok(resp,   add_special_tokens=False).input_ids

            ids = prompt_ids + resp_ids
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]

            # Build labels: ignore prompt with -100
            cut = min(len(prompt_ids), len(ids))
            labels = [-100]*cut + ids[cut:]

            self.examples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels":    torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

def pad_collate(tokenizer, max_len: int):
    pad_id = tokenizer.pad_token_id
    def _fn(batch):
        maxL = min(max_len, max(x["input_ids"].size(0) for x in batch))
        def pad_vec(v, fill):
            if v.size(0) >= maxL: return v[:maxL]
            out = torch.full((maxL,), fill, dtype=v.dtype)
            out[:v.size(0)] = v
            return out
        input_ids = torch.stack([pad_vec(x["input_ids"], pad_id) for x in batch])
        labels    = torch.stack([pad_vec(x["labels"], -100)   for x in batch])
        attn_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
    return _fn

def load_alpaca(tokenizer, cfg: Config):
    ds = load_dataset(cfg.dataset, split="train")  # only train split exists
    ds = ds.shuffle(seed=cfg.seed)
    n_total = len(ds)
    n_val = max(1, int(n_total * cfg.val_ratio))
    val = ds.select(range(n_val))
    trn = ds.select(range(n_val, n_total))

    train_ds = AlpacaSFTDataset(tokenizer, trn, cfg.max_seq_len)
    valid_ds = AlpacaSFTDataset(tokenizer, val, cfg.max_seq_len)
    return train_ds, valid_ds

# ===================== model & targets =====================
def _build_target_modules_llama(model, cfg: Config):
    # LLaMA family module names in HF:
    #   model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}
    #   model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}
    try:
        n_layers = int(model.config.num_hidden_layers)
    except Exception:
        n_layers = len(model.model.layers)
    start = max(0, n_layers - int(cfg.last_k_layers))

    want = set(list(cfg.target_projections))
    names = []
    for i in range(start, n_layers):
        attn = f"model.layers.{i}.self_attn"
        if "q" in want: names.append(f"{attn}.q_proj")
        if "k" in want: names.append(f"{attn}.k_proj")
        if "v" in want: names.append(f"{attn}.v_proj")
        if "o" in want: names.append(f"{attn}.o_proj")
        if cfg.include_mlp:
            mlp = f"model.layers.{i}.mlp"
            names += [f"{mlp}.gate_proj", f"{mlp}.up_proj", f"{mlp}.down_proj"]
    return names

def add_lora(model, cfg: Config):
    target_modules = _build_target_modules_llama(model, cfg)
    replaced = 0
    for name, lin in list(_iter_named_linears(model)):
        if not _matches_any(name, target_modules):
            continue
        d_out, d_in = _module_in_out(lin)
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRAWrapper(lin, d_in, d_out, r=cfg.rank, lora_alpha=cfg.lora_alpha))
        replaced += 1
    print(f"[LoRA] Wrapped {replaced} modules (rank={cfg.rank}, targets={cfg.target_projections}, include_mlp={cfg.include_mlp}).")

    # Train only adapters
    tparams = 0; allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A", "B"))
        p.requires_grad = req
        if req: tparams += p.numel()
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[Trainable LORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model

def add_bayeslora(model, cfg: Config):
    target_modules = _build_target_modules_llama(model, cfg)
    model = apply_bayeslora(
        model,
        target_modules=target_modules,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        kl_scale=cfg.kl_scale,
        tie_alpha_per_rank=True,
        local_reparam=True,
        freeze_base=True,
        exclude=["lm_head"],
    )
    tparams = 0; allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A_mu","B_mu","rank_log_sigma2","A_log_sigma2","B_log_sigma2"))
        p.requires_grad = req
        if req: tparams += p.numel()
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[BayesLoRA] targets={cfg.target_projections}, include_mlp={cfg.include_mlp}")
    print(f"[Trainable BAYESLORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model

def load_model_and_tokenizer(cfg: Config):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_dtype(cfg.dtype)

    # --- Optional: QLoRA base (massive VRAM savings) ---
    use_4bit = True  # set False if you don’t want quantization
    bnb_cfg = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            bnb_cfg = None
            print("[warn] bitsandbytes not available; falling back to full precision base.")

    kwargs = dict(
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=dtype,  # torch_dtype is deprecated; use dtype
    )

    # FlashAttention 2 if installed
    try:
        kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        pass

    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.config.use_cache = False

    # --- Key memory saver ---
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model.gradient_checkpointing_enable()

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try: print("device:", torch.cuda.get_device_name(0))
        except Exception: pass
    print("model device:", next(model.parameters()).device)

    return model, tokenizer

# ===================== eval: simple loss/perplexity (with causal shift) =====================
@torch.no_grad()
def evaluate_loss(cfg: Config, model, dl: DataLoader, mc_samples: int = 1):
    device = next(model.parameters()).device
    model.eval()

    # ---- Deterministic perplexity (with shift) ----
    tot_nll = 0.0; tot_tokens = 0
    set_bayeslora_sampling(model, False)
    for batch in dl:
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)            # [B, T]
        attn = batch["attention_mask"].to(device, non_blocking=True)

        logits = model(input_ids=x, attention_mask=attn).logits      # [B, T, V]
        shift_logits = logits[:, :-1, :]
        shift_labels = y[:, 1:]
        loss_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (shift_labels != -100).sum().item()
        tot_nll += loss_sum.item()
        tot_tokens += n_tokens
    det_ppl = math.exp(tot_nll / max(1, tot_tokens)) if tot_tokens > 0 else float("inf")

    # ---- MC perplexity (probability averaging, with shift + safe gather) ----
    if mc_samples > 1 and cfg.method == "bayeslora":
        tot_nll = 0.0; tot_tokens = 0
        for batch in dl:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)           # [B, T]
            attn = batch["attention_mask"].to(device, non_blocking=True)

            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                logits = model(input_ids=x, attention_mask=attn).logits  # [B, T, V]
                p = logits.softmax(-1)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)

            avg_p = (probs_sum / float(mc_samples)).clamp_min(1e-12)     # [B, T, V]
            avg_p = avg_p[:, :-1, :]                                     # shift probs
            shift_labels = y[:, 1:]                                      # [B, T-1]
            mask = (shift_labels != -100)

            idx_safe = torch.where(mask, shift_labels, torch.zeros_like(shift_labels))
            tok_probs = avg_p.gather(-1, idx_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            tok_probs = tok_probs.masked_fill(~mask, 1.0).clamp_min(1e-12)

            nll = -torch.log(tok_probs).masked_select(mask).sum()
            tot_nll += nll.item()
            tot_tokens += mask.sum().item()

        mc_ppl = math.exp(tot_nll / max(1, tot_tokens)) if tot_tokens > 0 else float("inf")
    else:
        mc_ppl = det_ppl

    return det_ppl, mc_ppl

@torch.no_grad()
def evaluate_token_metrics(cfg, model, dl, mc_samples: int = 1, n_bins: int = 15):
    """
    Returns: (tokacc_det, ece_det, tokacc_mc, ece_mc)
    - Confidence = max-softmax prob per token
    - Accuracy/ECE computed only on supervised tokens (labels != -100)
    - MC uses probability averaging across samples, then max
    - Uses causal shift consistently
    """
    import numpy as np
    device = next(model.parameters()).device
    model.eval()

    # ---------- Deterministic ----------
    set_bayeslora_sampling(model, False)
    det_confs, det_corr = [], []
    for batch in dl:
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)            # [B, T]
        attn = batch["attention_mask"].to(device, non_blocking=True)

        logits = model(input_ids=x, attention_mask=attn).logits      # [B, T, V]
        probs  = logits.softmax(-1)[:, :-1, :]                       # shift
        yl     = y[:, 1:]
        mask   = (yl != -100)                                        # [B, T-1]

        confs, preds = probs.max(dim=-1)                             # [B, T-1]
        det_confs.extend(confs.masked_select(mask).float().tolist())
        det_corr.extend((preds.eq(yl) & mask).masked_select(mask).int().tolist())

    tokacc_det = float(np.mean(det_corr)) if det_corr else 0.0
    ece_det    = expected_calibration_error(det_confs, det_corr, n_bins=n_bins)

    # ---------- MC (probability averaging) ----------
    if mc_samples > 1 and cfg.method == "bayeslora":
        mc_confs, mc_corr = [], []
        for batch in dl:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)        # [B, T]
            attn = batch["attention_mask"].to(device, non_blocking=True)

            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                logits = model(input_ids=x, attention_mask=attn).logits
                p = logits.softmax(-1)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)

            avg_p = (probs_sum / float(mc_samples)).clamp_min(1e-12)[:, :-1, :]  # shift
            yl    = y[:, 1:]
            mask  = (yl != -100)

            confs, preds = avg_p.max(dim=-1)
            mc_confs.extend(confs.masked_select(mask).float().tolist())
            mc_corr.extend((preds.eq(yl) & mask).masked_select(mask).int().tolist())

        tokacc_mc = float(np.mean(mc_corr)) if mc_corr else 0.0
        ece_mc    = expected_calibration_error(mc_confs, mc_corr, n_bins=n_bins)
    else:
        tokacc_mc, ece_mc = tokacc_det, ece_det

    return tokacc_det, ece_det, tokacc_mc, ece_mc

# ===================== optim & training =====================
def _optimizer_and_sched(model, cfg: Config, total_steps: int):
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    return opt, sched

def _rebuild_opt_and_sched(model, cfg: Config, total_steps: int, current_step: int):
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)
    try: sched.last_epoch = max(0, current_step - 1)
    except Exception: pass
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
    LOG_EVERY = max(1, getattr(cfg, "diag_every", 500) // 5)

    while global_step < total_steps:
        for batch in train_dl:
            if global_step >= total_steps: break

            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            # Always sample for BayesLoRA (no warmup)
            set_bayeslora_sampling(model, (cfg.method == "bayeslora"))

            logits = model(input_ids=x, attention_mask=attn).logits  # [B, T, V]

            # --- SHIFTED LM LOSS (ignore_index honored) ---
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # KL with β=1.0 from step 0
            kl_term = torch.tensor(0.0, device=device, dtype=torch.float32)
            if cfg.method == "bayeslora":
                denom = max(1, x.size(0))
                kl_term = bayeslora_step(model, beta=1.0) / float(denom)

            loss = (loss_ce / cfg.grad_accum) + (kl_term / cfg.grad_accum)
            loss.backward()
            accum += 1

            if accum >= cfg.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    dt = time.time() - last_t; last_t = time.time()
                    print(f"step {global_step}/{total_steps} | loss {float(loss.item()):.4f} | "
                          f"ce {float(loss_ce.item()):.4f} | kl {float(kl_term.item()):.6f} | "
                          f"{dt:.2f}s since last")

                if global_step % cfg.diag_every == 0:
                    det_ppl, mc_ppl = evaluate_loss(cfg, model, valid_dl, mc_samples=(cfg.mc_eval if cfg.method=="bayeslora" else 1))
                    tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
                        cfg, model, valid_dl, mc_samples=(cfg.mc_eval if cfg.method=="bayeslora" else 1), n_bins=15
                    )
                    print(f"[Diag] step {global_step} | "
                          f"ppl(det) {det_ppl:.2f} | ppl(MC,{cfg.mc_eval}) {mc_ppl:.2f} | "
                          f"tok-acc(det) {tokacc_det*100:.1f}% | ECE(det) {ece_det:.3f} | "
                          f"ECE(MC,{cfg.mc_eval}) {ece_mc:.3f}")

                # prune-as-you-go
                if (cfg.method == "bayeslora" and cfg.prune_every > 0 and
                    global_step >= (cfg.prune_start_step or 0) and
                    global_step % cfg.prune_every == 0):
                    before = count_adapter_params(model)
                    changed = bayeslora_prune(model, logalpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
                    after = count_adapter_params(model)
                    ranks = get_adapter_ranks(model)
                    print(f"[Prune@{global_step}] adapter params: {before} -> {after} | ranks: {ranks}")
                    if changed:
                        opt, sched = _rebuild_opt_and_sched(model, cfg, total_steps, current_step=global_step)

    set_bayeslora_sampling(model, False)
    return model

# ===================== main =====================
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned")
    p.add_argument("--method", type=str, choices=["lora","bayeslora"], required=True)
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--max-seq-len", type=int, default=1024)

    # adapters
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--target-projections", type=str, default="qvo")
    p.add_argument("--last-k-layers", type=int, default=24)
    p.add_argument("--include-mlp", action="store_true")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--kl-scale", type=float, default=1e-6)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)
    p.add_argument("--prune-every", type=int, default=200)
    p.add_argument("--min-ranks", type=int, default=0)
    p.add_argument("--prune-start-step", type=int, default=0)

    # train
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)

    # eval/diag
    p.add_argument("--diag-every", type=int, default=200)
    p.add_argument("--mc-eval", type=int, default=5)
    p.add_argument("--max-eval-tokens", type=int, default=200000)

    # misc
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    cfg = Config(
        model_id=a.model_id, dataset=a.dataset, method=a.method,
        val_ratio=a.val_ratio, max_seq_len=a.max_seq_len,
        rank=a.rank, target_projections=a.target_projections, last_k_layers=a.last_k_layers,
        include_mlp=bool(a.include_mlp), lora_alpha=a.lora_alpha,
        kl_scale=a.kl_scale, logalpha_thresh=a.logalpha_thresh,
        prune_every=a.prune_every, min_ranks=a.min_ranks, prune_start_step=a.prune_start_step,
        lr=a.lr, weight_decay=a.weight_decay, max_steps=a.max_steps, warmup_steps=a.warmup_steps,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        diag_every=a.diag_every, mc_eval=a.mc_eval, max_eval_tokens=a.max_eval_tokens,
        dtype=a.dtype, seed=a.seed
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

    train_ds, valid_ds = load_alpaca(tokenizer, cfg)
    pin = torch.cuda.is_available()
    collate = pad_collate(tokenizer, cfg.max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=2 if pin else 0, pin_memory=pin, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=1 if pin else 0, pin_memory=pin, collate_fn=collate)

    print(f"Train ex: {len(train_ds)} | Valid ex: {len(valid_ds)} | max_seq_len: {cfg.max_seq_len}")
    if cfg.method == "bayeslora":
        print(f"[BayesLoRA] Trainable params: {count_adapter_params(model):,}")

    model = train_one(cfg, model, train_dl, valid_dl)

    det_ppl, mc_ppl = evaluate_loss(cfg, model, valid_dl, mc_samples=(cfg.mc_eval if cfg.method=="bayeslora" else 1))
    print("\n=== Results (Alpaca-cleaned) ===")
    print(f"Method: {cfg.method}")
    print(f"Perplexity (det): {det_ppl:.3f} | Perplexity (MC, {cfg.mc_eval}): {mc_ppl:.3f}")

if __name__ == "__main__":
    main()