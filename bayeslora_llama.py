#!/usr/bin/env python3
"""
bayeslora_llama.py
BayesLoRA vs LoRA on multiple-choice tasks (ARC-Challenge, HellaSwag) with Llama-2 7B.

- Base is FROZEN.
- Supports 4-bit (QLoRA) via --load-4bit.
- BayesLoRA: KL warmup, prune-as-you-go (with gating), MC eval, ECE.
- KL normalized Molchanov-style: divide by dataset size N.
- Pre-tokenizes datasets for speed; caches BayesLoRA modules to avoid per-step scans.
- ✅ Train-time MC sampling (Molchanov-style) after --sample-warmup-steps.

Example:
  python bayeslora_llama.py \
    --method bayeslora \
    --datasets arc_challenge,hellaswag \
    --model-id meta-llama/Llama-2-7b-hf \
    --rank 64 --lora-alpha 64 \
    --max-steps 1500 --warmup-steps 100 \
    --kl-scale 0.05 --kl-freeze-steps 200 --beta-warmup-steps 800 \
    --prune-every 1000 --prune-start-step 1200 --beta-prune-threshold 0.6 \
    --logalpha-thresh 3.0 --min-ranks 8 \
    --sample-warmup-steps 0 \
    --mc-eval 3 \
    --batch-size 2 --grad-accum 8 \
    --max-length 256 \
    --load-4bit
"""
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

# ===== BayesLoRA glue (expects bayeslora/hf.py in your PYTHONPATH) =====
from bayeslora.hf import (
    apply_bayeslora_to_llama, set_bayeslora_sampling,
    bayeslora_prune, get_adapter_ranks, count_adapter_params,
    LoRAWrapper, _iter_named_linears, _matches_any, _get_parent_and_attr, _module_in_out
)

# ===================== config =====================
@dataclass
class Config:
    model_id: str
    method: str               # "lora" or "bayeslora"
    datasets: List[str]       # ["arc_challenge", "hellaswag"]

    # LoRA / BayesLoRA
    rank: int = 64
    last_k_layers: int = 12
    target_projections: str = 'qv'
    lora_alpha: int = 64
    lora_dropout: float = 0.0   # not used in this minimal wrapper
    kl_scale: float = 0.05
    kl_freeze_steps: int = 200
    beta_warmup_steps: int = 800
    logalpha_thresh: float = 3.0
    prune_every: int = 0
    min_ranks: int = 2
    prune_start_step: Optional[int] = None
    beta_prune_threshold: float = 0.0  # e.g., 0.6 to delay pruning until β is strong
    sample_warmup_steps: int = 0       # ✅ start MC sampling after this many optimizer steps

    # Train
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 1500
    warmup_steps: int = 100
    batch_size: int = 2
    grad_accum: int = 8
    max_length: int = 256

    # Eval
    mc_eval: int = 10
    ece_bins: int = 15
    max_eval_samples: Optional[int] = None

    # System
    seed: int = 42
    load_4bit: bool = False

# ===================== utilities =====================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def expected_calibration_error(confs, trues, n_bins=15):
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

def _merge_list_of_dicts(batch):
    out = {}
    for ex in batch:
        for k, v in ex.items():
            out.setdefault(k, []).append(v)
    return out

# ===================== data =====================
def load_arc_challenge() -> Dict[str, List[Dict]]:
    ds = load_dataset("ai2_arc", "ARC-Challenge")
    def _map(ex):
        q = ex.get("question", "")
        ch = ex.get("choices", {})
        texts = ch.get("text", None)
        labels = ch.get("label", None)
        ans = ex.get("answerKey", None)

        if not isinstance(q, str) or not isinstance(texts, list) or not isinstance(labels, list):
            return None
        if ans is None or not isinstance(ans, str) or len(ans) != 1:
            return None
        if any(not isinstance(t, str) for t in texts): return None
        if any(not isinstance(l, str) or len(l) != 1 for l in labels): return None

        try:
            idx = labels.index(ans)
        except ValueError:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            idx = letters.find(ans)

        if idx < 0 or idx >= len(texts):
            return None
        return {"stem": q, "choices": texts, "answer_idx": idx}
    out = {}
    for split in ["train", "validation", "test"]:
        tmp = []
        for ex in ds[split]:
            m = _map(ex)
            if m is not None:
                tmp.append(m)
        out[split] = tmp
    return out

def load_hellaswag() -> Dict[str, List[Dict]]:
    ds = load_dataset("hellaswag")
    def _map(ex):
        stem = ex.get("ctx", "")
        choices = ex.get("endings", [])
        label = ex.get("label", None)
        try:
            idx = int(label)
        except Exception:
            return None
        if not isinstance(stem, str): return None
        if not isinstance(choices, list) or len(choices) == 0: return None
        if idx < 0 or idx >= len(choices): return None
        if any(not isinstance(c, str) for c in choices): return None
        return {"stem": stem, "choices": choices, "answer_idx": idx}
    out = {}
    for split in ["train", "validation", "test"]:
        tmp = []
        for ex in ds[split]:
            m = _map(ex)
            if m is not None:
                tmp.append(m)
        out[split] = tmp
    return out

def get_datasets(names: List[str]) -> Dict[str, Dict[str, List[Dict]]]:
    out = {}
    for name in names:
        key = name.strip().lower()
        if key in ("arc", "arc_challenge", "arc-challenge"):
            out["arc_challenge"] = load_arc_challenge()
        elif key == "hellaswag":
            out["hellaswag"] = load_hellaswag()
        else:
            raise ValueError(f"Unknown dataset: {name}")
    return out

# ===================== formatting, pretokenization & batching =====================
PROMPT_TEMPLATE = """You are a multiple-choice assistant. Choose the single best answer.

Question:
{stem}

Choices:
{choices}

Answer with just the letter (A, B, C, ...).
"""

def format_mc_prompt(stem: str, choices: List[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])
    return PROMPT_TEMPLATE.format(stem=stem, choices=lines)

def pretokenize_pool(tokenizer, pool: List[Dict], device, max_length: int):
    """Return list of dicts with tensors to avoid per-step tokenization."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    texts = []
    n_choices = []
    ans_idx = []
    for ex in pool:
        stem = ex["stem"]
        choices = ex["choices"]
        lines = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])
        texts.append(PROMPT_TEMPLATE.format(stem=stem, choices=lines))
        n_choices.append(len(choices))
        ans_idx.append(int(ex["answer_idx"]))
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = []
    for i in range(len(pool)):
        out.append({
            "input_ids": input_ids[i],
            "attention_mask": attn[i],
            "answer_idx": ans_idx[i],
            "n_choices": n_choices[i],
        })
    return out

class EncodedCollator:
    pad_id: int
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    def __call__(self, batch):
        input_ids = nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_id
        )
        attn = nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        answer_idx = torch.tensor([b["answer_idx"] for b in batch], dtype=torch.long, device=input_ids.device)
        n_choices = torch.tensor([b["n_choices"] for b in batch], dtype=torch.long, device=input_ids.device)
        return input_ids, attn, answer_idx, n_choices

def make_letter_token_ids(tokenizer, max_choices: int, device):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ids = []
    for i in range(max_choices):
        tok = tokenizer.encode(" " + letters[i], add_special_tokens=False)
        ids.append(tok[-1])
    return torch.tensor(ids, device=device, dtype=torch.long)

# ===================== model =====================
def load_model_and_tokenizer(cfg: Config):
    # Use TF32 on Ampere+ for speed
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant = None
    if cfg.load_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.max_length

    model_kwargs = dict(device_map="auto", torch_dtype=dtype)
    if quant is not None:
        model_kwargs["quantization_config"] = quant

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)

    # Freeze base; adapters only
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer

# --- helper: build target module list (q/k/v/o over last-K layers) ---
def _build_target_modules(model, cfg):
    # Defaults (overridable via cfg without touching your dataclass)
    projections = getattr(cfg, "target_projections", "qv")   # "qv", "qkvo", etc.
    last_k      = int(getattr(cfg, "last_k_layers", 12))     # e.g., 8, 12
    # detect number of layers
    try:
        num_layers = int(model.config.num_hidden_layers)
    except Exception:
        # fallback for LLaMA-like models
        num_layers = len(getattr(model, "model").layers)

    # normalize projections
    want_q = "q" in projections
    want_k = "k" in projections
    want_v = "v" in projections
    want_o = "o" in projections

    start = max(0, num_layers - last_k)
    layer_ids = list(range(start, num_layers))

    names = []
    for i in layer_ids:
        stem = f"model.layers.{i}.self_attn"
        if want_q: names.append(f"{stem}.q_proj")
        if want_k: names.append(f"{stem}.k_proj")
        if want_v: names.append(f"{stem}.v_proj")
        if want_o: names.append(f"{stem}.o_proj")
    return names

def add_lora(model, cfg: Config):
    # Prefer flash-attn v2 if available (big speed win)
    try:
        if getattr(model.config, "attn_implementation", None) != "flash_attention_2":
            model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    target_modules = _build_target_modules(model, cfg)
    replaced = 0

    for name, lin in list(_iter_named_linears(model)):
        if not _matches_any(name, target_modules):
            continue
        d_out, d_in = _module_in_out(lin)
        parent, attr = _get_parent_and_attr(model, name)
        wrapper = LoRAWrapper(lin, d_in, d_out, r=cfg.rank, lora_alpha=cfg.lora_alpha)
        setattr(parent, attr, wrapper)
        replaced += 1

    print(f"[LoRA] Wrapped {replaced} modules "
          f"(rank={cfg.rank}, targets={getattr(cfg,'target_projections','qv')}, "
          f"last_k={getattr(cfg,'last_k_layers',12)}).")

    # Unfreeze adapter params only
    tparams = 0
    allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        if any(k in n for k in ("A", "B")):
            p.requires_grad = True
            tparams += p.numel()
        else:
            p.requires_grad = False
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[Trainable LORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model

def add_bayeslora(model, cfg: Config):
    # Prefer flash-attn v2 if available
    try:
        if getattr(model.config, "attn_implementation", None) != "flash_attention_2":
            model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    target_modules = _build_target_modules(model, cfg)

    # IMPORTANT: hf.py must not upcast the base layer dtype.
    # (Ensure you've applied the earlier fix: .to(device=dev) without dtype)
    model = apply_bayeslora_to_llama(
        model,
        target_modules=target_modules,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        kl_scale=cfg.kl_scale,        # we still scale manually in the loss, this is kept for compat
        tie_alpha_per_rank=True,
        local_reparam=getattr(cfg, "local_reparam", False),
        freeze_base=True,
        exclude=["lm_head"],
    )

    # Unfreeze only BayesLoRA adapter params
    tparams = 0
    allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A_mu","B_mu","rank_log_sigma2","A_log_sigma2","B_log_sigma2"))
        p.requires_grad = req
        if req: tparams += p.numel()

    pct = 100.0 * tparams / max(1, allparams)
    print(f"[BayesLoRA] targets={getattr(cfg,'target_projections','qv')}, "
          f"last_k={getattr(cfg,'last_k_layers',12)}")
    print(f"[Trainable BAYESLORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")

    # Enable non-reentrant gradient checkpointing if available (performance-friendly)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    return model

# ===================== training / eval =====================
def linear_beta_schedule(step, cfg: Config) -> float:
    if step <= cfg.kl_freeze_steps:
        return 0.0
    t = step - cfg.kl_freeze_steps
    return min(1.0, t / max(1, cfg.beta_warmup_steps))

def _optimizer_and_sched(model, cfg: Config, total_steps: int):
    adapter_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adapter_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    return opt, sched

def _cache_bayeslora_modules(model):
    return [m for m in model.modules() if hasattr(m, "kl") and callable(getattr(m, "kl"))]

def train_one(cfg: Config, model, tokenizer, ds_train_enc: List[Dict]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    dl = DataLoader(ds_train_enc, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=EncodedCollator(pad_id=tokenizer.pad_token_id))
    total_steps = cfg.max_steps
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)

    N = len(ds_train_enc)   # Molchanov-style normalization
    bl_modules = _cache_bayeslora_modules(model)

    prune_start = cfg.prune_start_step if cfg.prune_start_step is not None else (cfg.kl_freeze_steps + cfg.beta_warmup_steps)

    global_step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    for input_ids, attn, _, _ in dl:
        if global_step >= total_steps:
            break

        # ✅ Train-time MC sampling after warmup (Molchanov-style)
        do_sample = (cfg.method == "bayeslora") and (global_step >= cfg.sample_warmup_steps)
        set_bayeslora_sampling(model, do_sample)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        beta = 0.0
        kl_term = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        if cfg.method == "bayeslora":
            beta = linear_beta_schedule(global_step, cfg)
            # Sum KL across cached BayesLoRA modules (FP32) and normalize by N
            kl_sum = None
            for m in bl_modules:
                k = m.kl()
                kl_sum = k if kl_sum is None else (kl_sum + k)
            if kl_sum is None:
                kl_sum = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
            kl_term = (cfg.kl_scale * beta * kl_sum) / float(N)

        loss = (loss_ce / cfg.grad_accum) + (kl_term / cfg.grad_accum)

        loss.backward()
        accum += 1
        if accum >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            global_step += 1

            if global_step % 100 == 0:
                if cfg.method == "bayeslora":
                    print(f"step {global_step}/{total_steps} | loss {float(loss.item()):.4f} | "
                          f"ce {float(loss_ce.item()):.4f} | kl {float(kl_term.item()):.6f} | β {beta:.2f} | sample {int(do_sample)}")
                else:
                    print(f"step {global_step}/{total_steps} | loss {float(loss.item()):.4f} | ce {float(loss_ce.item()):.4f}")

            # prune-as-you-go (gated)
            if (cfg.method == "bayeslora" and cfg.prune_every > 0 and
                global_step >= prune_start and beta >= cfg.beta_prune_threshold and
                global_step % cfg.prune_every == 0):
                before = count_adapter_params(model)
                bayeslora_prune(model, logalpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
                after = count_adapter_params(model)
                ranks = get_adapter_ranks(model)
                print(f"[Prune@{global_step}] adapter params: {before} -> {after}  ranks: {ranks}")

    # After training, default to deterministic behavior
    set_bayeslora_sampling(model, False)
    return model

@torch.no_grad()
def evaluate_mc(cfg: Config, model, tokenizer, ds_eval_enc: List[Dict], mc_samples: int = 1):
    """
    Multiple-choice scoring by next-token probability of the correct letter.
    Uses pretokenized inputs; computes letter token ids once per pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dl = DataLoader(ds_eval_enc, batch_size=cfg.batch_size, shuffle=False,
                    collate_fn=EncodedCollator(pad_id=tokenizer.pad_token_id))

    # find max #choices across eval set to precompute letter ids once
    max_c_global = max(int(b["n_choices"]) for b in ds_eval_enc)
    ltids_global = make_letter_token_ids(tokenizer, max_c_global, device)

    def run_pass(sample_flag: bool, samples: int):
        set_bayeslora_sampling(model, sample_flag)
        correct = 0
        total = 0
        all_conf = []
        all_true = []
        for input_ids, attn, answer_idx, n_choices in dl:
            B = input_ids.size(0)
            if samples == 1:
                out = model(input_ids=input_ids, attention_mask=attn)
                last_logits = out.logits[torch.arange(B, device=device), attn.sum(dim=1)-1]  # [B, V]
                probs = last_logits.softmax(-1)
                for b in range(B):
                    k = int(n_choices[b].item())
                    ids = ltids_global[:k]
                    p = probs[b].index_select(-1, ids)
                    pred = int(torch.argmax(p).item())
                    gold = int(answer_idx[b].item())
                    correct += 1 if pred == gold else 0
                    total += 1
                    all_conf.append(float(p[gold].item()))
                    all_true.append(1 if pred == gold else 0)
            else:
                probs_sum = None
                for _ in range(samples):
                    out = model(input_ids=input_ids, attention_mask=attn)
                    last_logits = out.logits[torch.arange(B, device=device), attn.sum(dim=1)-1]
                    p = last_logits.softmax(-1)
                    probs_sum = p if probs_sum is None else (probs_sum + p)
                probs = probs_sum / float(samples)
                for b in range(B):
                    k = int(n_choices[b].item())
                    ids = ltids_global[:k]
                    p = probs[b].index_select(-1, ids)
                    pred = int(torch.argmax(p).item())
                    gold = int(answer_idx[b].item())
                    correct += 1 if pred == gold else 0
                    total += 1
                    all_conf.append(float(p[gold].item()))
                    all_true.append(1 if pred == gold else 0)

        acc = correct / max(1, total)
        ece = expected_calibration_error(all_conf, all_true, n_bins=cfg.ece_bins)
        return acc, ece

    acc_det, ece_det = run_pass(sample_flag=False, samples=1)
    acc_mc, ece_mc = (acc_det, ece_det)
    if cfg.method == "bayeslora" and mc_samples > 1:
        acc_mc, ece_mc = run_pass(sample_flag=True, samples=mc_samples)

    # Back to deterministic sampling after eval
    set_bayeslora_sampling(model, False)
    return acc_det, ece_det, acc_mc, ece_mc

# ===================== main =====================
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--method", type=str, choices=["lora","bayeslora"], required=True)
    p.add_argument("--datasets", type=str, default="arc_challenge,hellaswag")

    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--target-projections", type=str, default="qv",
               help='subset of "qkvo" to adapt, e.g. "qv", "qkvo"')
    p.add_argument("--last-k-layers", type=int, default=12,
               help="wrap only the last K transformer layers")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)

    p.add_argument("--kl-scale", type=float, default=0.05)
    p.add_argument("--kl-freeze-steps", type=int, default=200)
    p.add_argument("--beta-warmup-steps", type=int, default=800)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)
    p.add_argument("--prune-every", type=int, default=0)
    p.add_argument("--min-ranks", type=int, default=2)
    p.add_argument("--prune-start-step", type=int, default=None)
    p.add_argument("--beta-prune-threshold", type=float, default=0.0)
    p.add_argument("--sample-warmup-steps", type=int, default=0)   # ✅

    p.add_argument("--mc-eval", type=int, default=10)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--max-eval-samples", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-4bit", action="store_true", default=False)

    args = p.parse_args()
    cfg = Config(
        model_id=args.model_id,
        method=args.method,
        datasets=[s.strip() for s in args.datasets.split(",") if s.strip()],
        rank=args.rank,
        target_projections = args.target_projections,
        last_k_layers = args.last_k_layers,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        kl_scale=args.kl_scale,
        kl_freeze_steps=args.kl_freeze_steps,
        beta_warmup_steps=args.beta_warmup_steps,
        logalpha_thresh=args.logalpha_thresh,
        prune_every=args.prune_every,
        min_ranks=args.min_ranks,
        prune_start_step=args.prune_start_step,
        beta_prune_threshold=args.beta_prune_threshold,
        sample_warmup_steps=args.sample_warmup_steps,
        mc_eval=args.mc_eval,
        ece_bins=args.ece_bins,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        load_4bit=args.load_4bit,
    )
    return cfg

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(cfg)

    # attach adapters
    if cfg.method == "bayeslora":
        model = add_bayeslora(model, cfg)
    else:
        model = add_lora(model, cfg)

    # datasets
    dsets = get_datasets(cfg.datasets)
    train_pool = []
    eval_pool = []
    for _, splits in dsets.items():
        train_pool.extend(splits["train"])
        eval_pool.extend(splits.get("validation", splits.get("test", [])))
    if cfg.max_eval_samples is not None:
        eval_pool = eval_pool[:cfg.max_eval_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pool_enc = pretokenize_pool(tokenizer, train_pool, device, cfg.max_length)
    eval_pool_enc  = pretokenize_pool(tokenizer, eval_pool,  device, cfg.max_length)

    # Train
    model = train_one(cfg, model, tokenizer, train_pool_enc)

    # Evaluate (det + MC)
    acc_det, ece_det, acc_mc, ece_mc = evaluate_mc(
        cfg, model, tokenizer, eval_pool_enc,
        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1)
    )

    print("\n=== Results ===")
    print(f"Method: {cfg.method}")
    print(f"Deterministic  - Acc: {acc_det*100:.2f}% | ECE: {ece_det:.4f}")
    print(f"MC ({cfg.mc_eval} samples) - Acc: {acc_mc*100:.2f}% | ECE: {ece_mc:.4f}")

if __name__ == "__main__":
    main()