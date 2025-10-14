#!/usr/bin/env python3
# bayeslora.py
# BayesLoRA trainer/evaluator for HF causal LMs — wired for ARC-Challenge + OASST1 + Llama-2-7B.
# Requires bayeslora/hf.py

"""
Examples:

# --- OASST1 (SFT) ---
python bayeslora.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --dataset oasst1 --oasst1-split train --oasst1-val-ratio 0.02 \
  --method bayeslora --dtype bfloat16 --mc-eval 8 \
  --rank 32 --lora-alpha 32 --last-k-layers 12 --target-projections qvo --include-mlp \
  --block-size 1024 --batch-size 4 --grad-accum 16 --lr 1e-4 --weight-decay 0.01 \
  --kl-scale 1e-7 --kl-freeze-steps 1000 --beta-warmup-steps 3000 \
  --sample-warmup-steps 1000
"""

import argparse
import math
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ===== BayesLoRA glue (expects bayeslora/hf.py in PYTHONPATH) =====
from bayeslora.hf import (
    apply_bayeslora_to_llama, set_bayeslora_sampling,
    bayeslora_prune, get_adapter_ranks, count_adapter_params,
    LoRAWrapper, _iter_named_linears, _get_parent_and_attr, _module_in_out,
    bayeslora_step,
)

# --------- optional log to file (kept from your last snippet) ----------
import sys
log = open("output.log", "a", buffering=1)  # line-buffered
sys.stdout = sys.stderr = log

# ===================== config =====================
@dataclass
class Config:
    # Model / dataset
    model_id: str = "gpt2"
    dataset: str = "arc-c"  # arc-c | oasst1 | wikitext2 | wikitext103
    block_size: int = 256

    # Method
    method: str = "bayeslora"   # "lora" or "bayeslora"

    # Adapters
    rank: int = 16
    lora_alpha: int = 16
    last_k_layers: int = 6
    target_projections: str = "qvo"  # subset of qkvo; add 'm' or use include_mlp
    include_mlp: bool = False
    manual_targets: Optional[str] = None  # comma-separated dotted names to force targets

    # BayesLoRA
    kl_scale: float = 2e-4
    kl_freeze_steps: int = 500
    beta_warmup_steps: int = 1500
    logalpha_thresh: float = 3.0
    prune_every: int = 0
    min_ranks: int = 1
    prune_start_step: Optional[int] = None
    beta_prune_threshold: float = 0.0
    sample_warmup_steps: int = 0

    # Train
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 2000
    warmup_steps: int = 200
    batch_size: int = 8
    grad_accum: int = 8

    # Eval / diagnostics
    mc_eval: int = 5
    max_eval_tokens: Optional[int] = 200_000
    diag_every: int = 500
    diag_token_cap: int = 50_000

    # System
    seed: int = 42
    dtype: str = "bfloat16"
    load_in_4bit: bool = False

    # ARC
    arc_split: str = "validation"          # "train" | "validation" | "test"
    arc_answer_format: str = "letter"      # "letter" | "option_text"
    arc_fewshot: int = 0
    arc_max_choices: int = 4
    arc_train: bool = False
    arc_train_mode: str = "lm"             # "lm" | "mc"

    # OASST1
    oasst1_split: str = "train"            # "train" (official split). We'll self-split for validation if needed.
    oasst1_val_ratio: float = 0.02         # if no validation split, carve from train
    oasst1_lang: Optional[str] = None      # e.g., "en" to filter; None = all
    oasst1_max_examples: Optional[int] = None  # cap for quick tests


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
    import numpy as np
    confs = np.asarray(confs, dtype=np.float32).reshape(-1)
    trues = np.asarray(trues, dtype=np.int32).reshape(-1)
    ok = np.isfinite(confs)
    confs = confs[ok]
    trues = trues[:len(ok)][ok] if len(trues) != len(ok) else trues[ok]
    m = min(len(confs), len(trues))
    if m == 0:
        return 0.0
    confs = confs[:m]; trues = trues[:m]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi)
        if not np.any(mask): continue
        acc = trues[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def _clean(s: str) -> str:
    return (s or "").strip()

# ===================== data: tokenize -> concat -> chunk =====================
class BlockDataset(Dataset):
    def __init__(self, ids: torch.Tensor, block_size: int):
        assert ids.dim() == 1
        n_blocks = (ids.numel() - 1) // block_size
        usable = n_blocks * block_size + 1
        ids = ids[:usable]
        self.x = ids[:-1].view(n_blocks, block_size)
        self.y = ids[1:].view(n_blocks, block_size)

    def __len__(self): return self.x.size(0)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def load_wikitext2(tokenizer, block_size: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tok_concat(split):
        texts = "\n\n".join(ds[split]["text"])
        enc = tokenizer(texts, return_tensors=None, add_special_tokens=False)
        return torch.tensor(enc["input_ids"], dtype=torch.long)
    return BlockDataset(tok_concat("train"), block_size), BlockDataset(tok_concat("validation"), block_size)

def load_wikitext103(tokenizer, block_size: int):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    def tok_concat(split):
        texts = "\n\n".join(ds[split]["text"])
        enc = tokenizer(texts, return_tensors=None, add_special_tokens=False)
        return torch.tensor(enc["input_ids"], dtype=torch.long)
    return BlockDataset(tok_concat("train"), block_size), BlockDataset(tok_concat("validation"), block_size)

# ===================== OASST1 helpers =====================
OASST_PROMPT_TMPL = "### Instruction:\n{prompt}\n\n### Response:\n"

class OASST1Dataset(Dataset):
    """
    Builds (x, y) where:
      x = tokenized prompt + '### Response:\n' + assistant,
      y = -100 for prompt segment, tokens for assistant segment (mask-safe CE).
    Pairs are built by matching assistant nodes to their prompter parent.
    """
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        block_size: int = 1024,
        lang: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.02,
        role_key: str = "role",
        text_key: str = "text",
        parent_key: str = "parent_id",
        id_key: str = "message_id",
        lang_key: str = "lang",
    ):
        super().__init__()
        rng = random.Random(seed)

        # Load raw dataset (OASST1 generally ships as a single 'train' split)
        raw = load_dataset("OpenAssistant/oasst1")
        if split not in raw:
            # Self split from 'train' if needed
            base = raw["train"]
            idxs = list(range(len(base)))
            rng.shuffle(idxs)
            cut = int(len(base) * (1.0 - val_ratio))
            train_idxs = idxs[:cut]
            val_idxs = idxs[cut:]
            raw = DatasetDict({
                "train": base.select(train_idxs),
                "validation": base.select(val_idxs),
            })

        ds = raw[split]

        # Map id -> record for quick parent lookup
        id2rec: Dict[str, dict] = {}
        for rec in ds:
            rid = rec.get(id_key)
            if rid: id2rec[rid] = rec

        # Build prompt/assistant pairs
        pairs = []
        for rec in ds:
            if rec.get(role_key) != "assistant":
                continue
            p_id = rec.get(parent_key)
            if not p_id:  # orphan assistant (rare)
                continue
            parent = id2rec.get(p_id)
            if not parent or parent.get(role_key) != "prompter":
                continue
            if lang and (parent.get(lang_key) != lang or rec.get(lang_key) != lang):
                continue
            user = _clean(parent.get(text_key, ""))
            assistant = _clean(rec.get(text_key, ""))
            if not user or not assistant:
                continue
            pairs.append((user, assistant))

        if max_examples is not None:
            pairs = pairs[: int(max_examples)]

        # Tokenize pairs into fixed-length blocks with mask-safe labels
        self.pad_id = tokenizer.pad_token_id
        self.block_size = int(block_size)
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for user, assistant in pairs:
            prompt_prefix = OASST_PROMPT_TMPL.format(prompt=user)
            prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
            resp_ids   = tokenizer(assistant,      add_special_tokens=False)["input_ids"]

            ids = prompt_ids + resp_ids
            labels = ([-100] * len(prompt_ids)) + resp_ids

            # left-truncate then left-pad to block_size
            ids = ids[-self.block_size:]
            labels = labels[-self.block_size:]
            if len(ids) < self.block_size:
                pad_len = self.block_size - len(ids)
                ids = [self.pad_id] * pad_len + ids
                labels = [-100] * pad_len + labels

            self.examples.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

def pad_to_block_collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

# ===================== ARC helpers (unchanged functional behavior) =====================
def load_arc_challenge():
    return load_dataset("ai2_arc", "ARC-Challenge")

def build_arc_prompt(q, choices, fewshots=None):
    parts = []
    if fewshots:
        parts.append("\n\n".join(
            "Solve the multiple-choice question.\n\nQ: {q}\n{opts}\nAnswer: {a}".format(
                q=fq, opts="\n".join([f"{l}) {t}" for l,t in fc]), a=fa
            ) for (fq, fc, fa) in fewshots
        ))
    body = "Solve the multiple-choice question.\n\n"
    body += f"Q: {q}\n"
    body += "\n".join([f"{l}) {t}" for l, t in choices])
    body += "\nAnswer:"
    parts.append(body)
    return "\n\n".join(parts)

def _subset_choices(labels, texts, max_choices, answer_key):
    if max_choices is None or max_choices <= 0 or len(labels) <= max_choices:
        return labels, texts
    if answer_key in labels[:max_choices]:
        return labels[:max_choices], texts[:max_choices]
    if answer_key in labels:
        gi = labels.index(answer_key)
        start = max(0, min(gi - (max_choices - 1), len(labels) - max_choices))
        end = start + max_choices
        return labels[start:end], texts[start:end]
    return labels[:max_choices], texts[:max_choices]

def arc_build_batch(tokenizer, items, answer_format="option_text", fewshot_k=0, max_choices=4):
    batch_prompts, batch_opts = [], []
    for it in items:
        q = _clean(it["question"])
        raw_labels = [str(x) for x in it["choices"]["label"]]
        raw_texts  = [_clean(str(x)) for x in it["choices"]["text"]]
        ans = _clean(it["answerKey"])

        labels, texts = _subset_choices(raw_labels, raw_texts, max_choices, ans)
        pairs = list(zip(labels, texts))

        prompt = build_arc_prompt(q, pairs, fewshots=None)
        batch_prompts.append(prompt)

        opts_tok = []
        if answer_format == "letter":
            for l, _ in pairs:
                ids = tokenizer(l, add_special_tokens=False, return_tensors=None)["input_ids"]
                opts_tok.append(torch.tensor(ids, dtype=torch.long))
        else:
            for _, t in pairs:
                ids = tokenizer(" " + t, add_special_tokens=False, return_tensors=None)["input_ids"]
                opts_tok.append(torch.tensor(ids, dtype=torch.long))
        batch_opts.append(opts_tok)
    return batch_prompts, batch_opts

@torch.no_grad()
def score_arc_batch(model, tokenizer, batch_prompts, batch_options_tok, mc_samples=1):
    device = next(model.parameters()).device
    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    B = input_ids.size(0)
    K = max(len(opts) for opts in batch_options_tok)

    def _score_once(sampled: bool):
        set_bayeslora_sampling(model, sampled)
        scores = torch.full((B, K), float("-inf"), device=device, dtype=torch.float32)
        pad_id = tokenizer.pad_token_id
        for k in range(K):
            seqs, lens, optLs = [], [], []
            for i in range(B):
                if k >= len(batch_options_tok[i]): seqs.append(None); lens.append(0); optLs.append(0); continue
                opt_ids = batch_options_tok[i][k].to(device)
                seq = torch.cat([input_ids[i], opt_ids], dim=0)
                seqs.append(seq); lens.append(seq.size(0)); optLs.append(opt_ids.size(0))
            maxL = max(lens)
            batch = torch.full((B, maxL), pad_id, dtype=torch.long, device=device)
            mask  = torch.zeros((B, maxL), dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                if s is None: continue
                L = s.size(0)
                batch[i, :L] = s
                mask[i, :L] = 1
            out = model(input_ids=batch, attention_mask=mask)
            logits = out.logits
            sumlogp = torch.zeros(B, device=device, dtype=torch.float32)
            for i in range(B):
                if seqs[i] is None: continue
                optL = optLs[i]; L = lens[i]; start = L - optL
                lp = F.log_softmax(logits[i, start-1:L-1, :], dim=-1)
                tgt = batch[i, start:L]
                sumlogp[i] = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
            scores[:, k] = sumlogp
        return scores

    det_scores = _score_once(sampled=False)
    if mc_samples <= 1:
        return det_scores, det_scores
    prob_sum = torch.zeros_like(det_scores, dtype=torch.float32, device=device)
    for _ in range(mc_samples):
        s = _score_once(sampled=True)
        prob_sum += torch.softmax(s, dim=-1)
    set_bayeslora_sampling(model, False)
    mc_scores = torch.log(prob_sum / float(mc_samples) + 1e-12)
    return det_scores, mc_scores

@torch.no_grad()
def evaluate_arc(cfg: Config, model, tokenizer, split="validation"):
    ds = load_arc_challenge()[split]
    items = [dict(
        question=_clean(ex["question"]),
        choices={"label": ex["choices"]["label"], "text": ex["choices"]["text"]},
        answerKey=_clean(ex["answerKey"])
    ) for ex in ds]

    batch_prompts, batch_opts = arc_build_batch(
        tokenizer, items,
        answer_format=cfg.arc_answer_format,
        fewshot_k=cfg.arc_fewshot,
        max_choices=cfg.arc_max_choices,
    )

    B = len(items)
    bs = 8
    det_chunks, mc_chunks, gold = [], [], []
    for i in range(0, B, bs):
        j = min(B, i + bs)
        det, mc = score_arc_batch(
            model, tokenizer,
            batch_prompts[i:j], batch_opts[i:j],
            mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1)
        )
        det_chunks.append(det.cpu()); mc_chunks.append(mc.cpu())
        gold.extend([ord(items[k]["answerKey"][0]) - ord('A') for k in range(i, j)])

    import numpy as np
    det = torch.cat(det_chunks, dim=0).numpy()
    mc  = torch.cat(mc_chunks,  dim=0).numpy()
    gold = np.array(gold, dtype=np.int32)

    det_pred = det.argmax(-1); mc_pred = mc.argmax(-1)
    det_acc = float((det_pred == gold).mean())
    mc_acc  = float((mc_pred  == gold).mean())

    def _ece_from_logits(logits, labels, n_bins=15):
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        confs = probs.max(-1); trues = (probs.argmax(-1) == labels).astype(np.int32)
        bins = np.linspace(0.0, 1.0, n_bins+1); e=0.0
        for b in range(n_bins):
            lo,hi=bins[b],bins[b+1]; m=(confs>lo)&(confs<=hi)
            if not m.any(): continue
            acc = trues[m].mean(); cf = confs[m].mean()
            e += m.mean() * abs(acc-cf)
        return float(e)

    det_ece = _ece_from_logits(det, gold); mc_ece = _ece_from_logits(mc, gold)
    print(f"[ARC-{split}] acc(det)={det_acc*100:.2f}% | acc(MC,{cfg.mc_eval})={mc_acc*100:.2f}% | "
          f"ECE(det)={det_ece:.3f} | ECE(MC)={mc_ece:.3f}")
    return det_acc, mc_acc, det_ece, mc_ece

# ===================== eval helpers (LM) =====================
@torch.no_grad()
def evaluate_perplexity(cfg: Config, model, dl: DataLoader, mc_samples: int = 1, current_step: Optional[int] = None) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    total_tokens = 0; nll_det = 0.0; nll_mc = 0.0

    set_bayeslora_sampling(model, False)
    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(input_ids=x).logits
        mask = (y != -100)
        if mask.any():
            lp = F.log_softmax(logits, dim=-1)
            idx = y.clamp_min(0)
            tok_lp = lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            nll_det += (-tok_lp[mask]).sum().item()
            total_tokens += int(mask.sum().item())
    ppl_det = math.exp(nll_det / max(1, total_tokens)) if total_tokens > 0 else float("inf")

    mc_allowed = (mc_samples > 1 and cfg.method == "bayeslora" and current_step is None)
    if mc_allowed:
        processed = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                p = model(input_ids=x).logits.softmax(-1)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)
            avg_probs = (probs_sum / float(mc_samples)).clamp_min(1e-12)
            mask = (y != -100)
            if mask.any():
                idx = y.clamp_min(0)
                tok_p = avg_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                nll_mc += (-(tok_p[mask]).log()).sum().item()
                processed += int(mask.sum().item())
        ppl_mc = math.exp(nll_mc / max(1, processed)) if processed > 0 else ppl_det
    else:
        ppl_mc = ppl_det
    return ppl_det, ppl_mc

@torch.no_grad()
def evaluate_token_metrics(cfg: Config, model, dl: DataLoader, mc_samples: int = 1, token_cap: Optional[int] = None):
    import numpy as np
    import torch as _torch
    device = next(model.parameters()).device
    model.eval()

    def _gather_valid_confs_trues(probs, y):
        mask = (y != -100)
        if not mask.any():
            return _torch.empty(0, dtype=_torch.float32), _torch.empty(0, dtype=_torch.int32)
        confs, preds = probs.max(dim=-1)
        trues = (preds == y) & mask
        return (
            confs[mask].detach().to(dtype=_torch.float32, device="cpu"),
            trues.int().detach().to(dtype=_torch.int32, device="cpu"),
        )

    set_bayeslora_sampling(model, False)
    det_confs_list, det_trues_list = [], []
    counted = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        probs = model(input_ids=x).logits.softmax(-1)
        confs_cpu, trues_cpu = _gather_valid_confs_trues(probs, y)
        if token_cap is not None:
            remaining = max(0, token_cap - counted)
            if remaining == 0: break
            if confs_cpu.numel() > remaining:
                confs_cpu = confs_cpu[:remaining]; trues_cpu = trues_cpu[:remaining]
        det_confs_list.append(confs_cpu); det_trues_list.append(trues_cpu)
        counted += confs_cpu.numel()
        if token_cap is not None and counted >= token_cap: break

    if det_confs_list:
        det_confs_cat = _torch.cat(det_confs_list, dim=0).numpy()
        det_trues_cat = _torch.cat(det_trues_list, dim=0).numpy()
        det_acc = float(det_trues_cat.mean()); ece_det = expected_calibration_error(det_confs_cat.tolist(), det_trues_cat.tolist(), n_bins=15)
    else:
        det_acc, ece_det = 0.0, 0.0

    if mc_samples > 1 and cfg.method == "bayeslora":
        mc_confs_list, mc_trues_list = [], []
        counted = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            probs_sum = None
            for _ in range(mc_samples):
                set_bayeslora_sampling(model, True)
                p = model(input_ids=x).logits.softmax(-1)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            set_bayeslora_sampling(model, False)
            probs = probs_sum / float(mc_samples)
            confs_cpu, trues_cpu = _gather_valid_confs_trues(probs, y)
            if token_cap is not None:
                remaining = max(0, token_cap - counted)
                if remaining == 0: break
                if confs_cpu.numel() > remaining:
                    confs_cpu = confs_cpu[:remaining]; trues_cpu = trues_cpu[:remaining]
            mc_confs_list.append(confs_cpu); mc_trues_list.append(trues_cpu)
            counted += confs_cpu.numel()
            if token_cap is not None and counted >= token_cap: break

        if mc_confs_list:
            mc_confs_cat = _torch.cat(mc_confs_list, dim=0).numpy()
            mc_trues_cat = _torch.cat(mc_trues_list, dim=0).numpy()
            tokacc_mc = float(mc_trues_cat.mean()); ece_mc = expected_calibration_error(mc_confs_cat.tolist(), mc_trues_cat.tolist(), n_bins=15)
        else:
            tokacc_mc, ece_mc = det_acc, ece_det
    else:
        tokacc_mc, ece_mc = det_acc, ece_det

    return det_acc, ece_det, tokacc_mc, ece_mc

# ===================== model & adapters =====================
_LAYER_INDEX_PATTERNS = [
    r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.decoder\.layers\.(\d+)\.", r"gpt_neox\.layers\.(\d+)\.", r"\.layer\.(\d+)\.",
]
_ATT_TOKENS = ["q_proj","k_proj","v_proj","o_proj","c_attn","c_proj","Wqkv","Wo","query_key_value","dense","dense_h_to_4h","dense_4h_to_h","attn.proj","attention.dense"]
_MLP_TOKENS = ["gate_proj","up_proj","down_proj","w1","w2","w3","fc_in","fc_out","dense_h_to_4h","dense_4h_to_h"]

def _extract_layer_index(name: str) -> Optional[int]:
    for pat in _LAYER_INDEX_PATTERNS:
        m = re.search(pat, name)
        if m:
            try: return int(m.group(1))
            except Exception: pass
    return None

def _want_token(token: str, cfg: Config) -> bool:
    tproj = cfg.target_projections.lower()
    want_q = "q" in tproj; want_k = "k" in tproj; want_v = "v" in tproj
    want_o = ("o" in tproj) or ("p" in tproj); want_m = cfg.include_mlp or ("m" in tproj)
    if token in ("q_proj","c_attn","Wqkv","query_key_value"): return want_q or want_v or want_k
    if token in ("k_proj",): return want_k
    if token in ("v_proj",): return want_v
    if token in ("o_proj","c_proj","Wo","attn.proj","attention.dense","dense"): return want_o
    if token in _MLP_TOKENS: return want_m
    return False

def discover_target_modules(model: nn.Module, cfg: Config) -> List[str]:
    if cfg.manual_targets:
        return [t.strip() for t in cfg.manual_targets.split(",") if t.strip()]
    candidates: List[str] = []
    for name, _ in _iter_named_linears(model):
        lower = name.lower()
        for tok in _ATT_TOKENS + _MLP_TOKENS:
            if tok in lower and _want_token(tok, cfg):
                candidates.append(name); break
    if not candidates:
        print("[BayesLoRA] WARNING: auto-discovery found no targets. Consider --manual-targets.")
        return []
    try:
        n_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", None) or None
    except Exception:
        n_layers = None
    if n_layers is None:
        max_idx = -1
        for n in candidates:
            idx = _extract_layer_index(n)
            if idx is not None: max_idx = max(max_idx, idx)
        n_layers = (max_idx + 1) if max_idx >= 0 else None
    if n_layers is None or cfg.last_k_layers <= 0:
        return candidates
    start = max(0, int(n_layers) - int(cfg.last_k_layers))
    filtered = []
    for n in candidates:
        idx = _extract_layer_index(n)
        if idx is None or idx >= start:
            filtered.append(n)
    return filtered

def load_model_and_tokenizer(cfg: Config):
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9

    dtype = get_dtype(cfg.dtype)
    bnb_cfg = None
    if cfg.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes not available")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16),
        )

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, device_map="auto", torch_dtype=dtype, quantization_config=bnb_cfg,
        )
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id)

    for p in model.parameters():
        p.requires_grad = False

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    print("model device:", next(model.parameters()).device)
    return model, tokenizer

def add_lora(model, cfg: Config):
    target_modules = discover_target_modules(model, cfg)
    replaced = 0
    for name, lin in list(_iter_named_linears(model)):
        if name not in target_modules: continue
        d_out, d_in = _module_in_out(lin)
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRAWrapper(lin, d_in, d_out, r=cfg.rank, lora_alpha=cfg.lora_alpha))
        replaced += 1
    print(f"[LoRA] Wrapped {replaced} modules (rank={cfg.rank}, last_k={cfg.last_k_layers}, targets={cfg.target_projections}).")

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
    target_modules = discover_target_modules(model, cfg)
    model = apply_bayeslora_to_llama(
        model, target_modules=target_modules, rank=cfg.rank, lora_alpha=cfg.lora_alpha,
        kl_scale=cfg.kl_scale, tie_alpha_per_rank=False, local_reparam=True,
        freeze_base=True, exclude=["lm_head", "embed_tokens"],
    )
    tparams = 0; allparams = 0
    for n, p in model.named_parameters():
        allparams += p.numel()
        req = any(k in n for k in ("A_mu","B_mu","rank_log_sigma2","A_log_sigma2","B_log_sigma2"))
        p.requires_grad = req
        if req: tparams += p.numel()
    pct = 100.0 * tparams / max(1, allparams)
    print(f"[BayesLoRA] targets={cfg.target_projections}, last_k={cfg.last_k_layers}")
    print(f"[Trainable BAYESLORA] {tparams:,}/{allparams:,} params ({pct:.3f}%)")
    return model

# ===================== training =====================
def _optimizer_and_sched(model, cfg: Config, total_steps: int):
    adapter_params = [p for _, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adapter_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    return opt, sched

def _rebuild_opt_and_sched(model, cfg: Config, total_steps: int, current_step: int):
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)
    try: sched.last_epoch = max(0, current_step - 1)
    except Exception: pass
    return opt, sched

def train_one(cfg: Config, model, tokenizer, train_dl: DataLoader, valid_dl: DataLoader):
    import time
    device = next(model.parameters()).device
    model.train()

    total_steps = cfg.max_steps
    opt, sched = _optimizer_and_sched(model, cfg, total_steps)

    batches_per_epoch = len(train_dl)
    steps_per_epoch = math.ceil(batches_per_epoch / max(1, cfg.grad_accum))
    print(f"[Train] batches/epoch={batches_per_epoch} | grad_accum={cfg.grad_accum} | "
          f"~steps/epoch={steps_per_epoch} | max_steps={total_steps} | diag_every={cfg.diag_every}")

    global_step = 0; accum = 0; opt.zero_grad(set_to_none=True)
    last_t = time.time(); LOG_EVERY = max(1, getattr(cfg, "diag_every", 500) // 5)

    while global_step < total_steps:
        for x, y in train_dl:
            if global_step >= total_steps: break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            do_sample = (cfg.method == "bayeslora") and (global_step >= cfg.sample_warmup_steps)
            set_bayeslora_sampling(model, do_sample)

            logits = model(input_ids=x).logits
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            valid = (y_flat != -100)
            if not torch.any(valid): continue

            loss_ce = F.cross_entropy(logits_flat[valid], y_flat[valid], reduction="mean")

            kl_term = torch.tensor(0.0, device=device, dtype=torch.float32)
            if cfg.method == "bayeslora":
                beta = 0.0
                if global_step > cfg.kl_freeze_steps:
                    beta = min(1.0, (global_step - cfg.kl_freeze_steps) / max(1, cfg.beta_warmup_steps))
                tokens = float(x.size(0) * x.size(1))
                kl_term = bayeslora_step(model, beta=beta) / tokens
            else:
                beta = 0.0

            loss = (loss_ce / cfg.grad_accum) + (kl_term / cfg.grad_accum)
            loss.backward(); accum += 1

            if accum >= cfg.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                accum = 0; global_step += 1

                if global_step % LOG_EVERY == 0:
                    dt = time.time() - last_t; last_t = time.time()
                    print(f"step {global_step}/{total_steps} | loss {float(loss.item()):.4f} | "
                          f"ce {float(loss_ce.item()):.4f} | kl {float(kl_term.item()):.6f} | β {beta:.2f} | "
                          f"{dt:.2f}s since last")

                if global_step % cfg.diag_every == 0:
                    ppl_det, ppl_mc = evaluate_perplexity(
                        cfg, model, valid_dl,
                        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
                        current_step=global_step,
                    )
                    tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
                        cfg, model, valid_dl,
                        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
                        token_cap=cfg.diag_token_cap
                    )
                    mc_active = (cfg.method=="bayeslora")
                    print(f"[Diag] step {global_step}/{total_steps} | ppl(det) {ppl_det:.2f} | "
                          f"ppl({'MC' if mc_active else 'det'},{cfg.mc_eval}) {ppl_mc:.2f} | "
                          f"tok-acc(det) {tokacc_det*100:.1f}% | ECE(det) {ece_det:.3f} | "
                          f"ECE({'MC' if mc_active else 'det'}) {ece_mc:.3f}")

                    # Optional: fast ARC probe (harmless if dataset != arc-c)
                    try:
                        det_arc_acc, mc_arc_acc, n_arc = evaluate_arc_quick(
                            cfg, model, tokenizer, split=cfg.arc_split,
                            max_items=64, allow_mc=mc_active
                        )
                        print(f"[Diag-ARC] step {global_step} | MC acc(det) {det_arc_acc*100:.1f}% | "
                              f"MC acc({'MC' if mc_active else 'det'}) {mc_arc_acc*100:.1f}% | n={n_arc}")
                    except Exception:
                        pass

                if (cfg.method == "bayeslora" and cfg.prune_every > 0 and
                    global_step >= (cfg.prune_start_step if cfg.prune_start_step is not None else (cfg.kl_freeze_steps + cfg.beta_warmup_steps)) and
                    beta >= cfg.beta_prune_threshold and
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

# ---- ARC MC training utilities (unchanged) ----
def _tokize_opts(tokenizer, opts_text): return [torch.tensor(tokenizer(" " + t, add_special_tokens=False)["input_ids"], dtype=torch.long) for t in opts_text]
def build_arc_item(tokenizer, ex, max_choices=4):
    q = _clean(ex["question"]); raw_labels = [str(x) for x in ex["choices"]["label"]]
    raw_texts  = [_clean(str(x)) for x in ex["choices"]["text"]]; ans = _clean(ex["answerKey"])
    labels, texts = _subset_choices(raw_labels, raw_texts, max_choices, ans); pairs = list(zip(labels, texts))
    prompt = build_arc_prompt(q, pairs, fewshots=None); gold_idx = labels.index(ans)
    prompt_ids = torch.tensor(tokenizer(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long)
    return {"prompt_ids": prompt_ids, "choices_text": texts, "gold_idx": gold_idx}
class ARCItemDataset(Dataset):
    def __init__(self, tokenizer, split, max_choices=4):
        ds = load_arc_challenge()[split]
        self.items = [build_arc_item(tokenizer, ex, max_choices=max_choices) for ex in ds]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
def arc_item_collate(batch): return batch
def arc_mc_step(model, tokenizer, batch_items, sample_at_train=True):
    device = next(model.parameters()).device; B = len(batch_items); K = max(len(it["choices_text"]) for it in batch_items)
    batch_opts_tok = [_tokize_opts(tokenizer, it["choices_text"]) for it in batch_items]
    pad = tokenizer.pad_token_id; set_bayeslora_sampling(model, bool(sample_at_train))
    scores = torch.zeros(B, K, device=device)
    for k in range(K):
        seqs, lens, optLs = [], [], []
        for i,it in enumerate(batch_items):
            if k >= len(batch_opts_tok[i]): seqs.append(None); lens.append(0); optLs.append(0); continue
            p = it["prompt_ids"].to(device); o = batch_opts_tok[i][k].to(device); s = torch.cat([p, o], dim=0)
            seqs.append(s); lens.append(s.size(0)); optLs.append(o.size(0))
        maxL = max(lens)
        batch = torch.full((B, maxL), pad, dtype=torch.long, device=device)
        mask  = torch.zeros((B, maxL), dtype=torch.long, device=device)
        for i,s in enumerate(seqs):
            if s is None: continue
            L = s.size(0); batch[i,:L]=s; mask[i,:L]=1
        logits = model(input_ids=batch, attention_mask=mask).logits
        for i in range(B):
            if seqs[i] is None: continue
            L = lens[i]; optL = optLs[i]; start = L - optL
            lp = F.log_softmax(logits[i, start-1:L-1, :], dim=-1)
            tgt = batch[i, start:L]
            scores[i,k] = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
    gold = torch.tensor([it["gold_idx"] for it in batch_items], dtype=torch.long, device=device)
    loss = F.cross_entropy(scores, gold); acc  = (scores.argmax(-1) == gold).float().mean().detach()
    set_bayeslora_sampling(model, False); return loss, acc

# ===================== CLI & main =====================
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--dataset", type=str, default="arc-c", help="arc-c | oasst1 | wikitext2 | wikitext103")
    p.add_argument("--method", type=str, choices=["lora","bayeslora"], required=True)

    # adapters
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--last-k-layers", type=int, default=6)
    p.add_argument("--target-projections", type=str, default="qvo")
    p.add_argument("--include-mlp", action="store_true")
    p.add_argument("--manual-targets", type=str, default=None)

    # BayesLoRA
    p.add_argument("--kl-scale", type=float, default=2e-4)
    p.add_argument("--kl-freeze-steps", type=int, default=500)
    p.add_argument("--beta-warmup-steps", type=int, default=1500)
    p.add_argument("--logalpha-thresh", type=float, default=3.0)
    p.add_argument("--prune-every", type=int, default=0)
    p.add_argument("--min-ranks", type=int, default=1)
    p.add_argument("--prune-start-step", type=int, default=None)
    p.add_argument("--beta-prune-threshold", type=float, default=0.0)
    p.add_argument("--sample-warmup-steps", type=int, default=0)

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

    # system
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # ARC
    p.add_argument("--arc-split", type=str, default="validation", choices=["train","validation","test"])
    p.add_argument("--arc-answer-format", type=str, default="letter", choices=["letter","option_text"])
    p.add_argument("--arc-fewshot", type=int, default=0)
    p.add_argument("--arc-max-choices", type=int, default=4)
    p.add_argument("--arc-train", action="store_true")
    p.add_argument("--arc-train-mode", type=str, default="lm", choices=["lm","mc"])

    # OASST1
    p.add_argument("--oasst1-split", type=str, default="train")
    p.add_argument("--oasst1-val-ratio", type=float, default=0.02)
    p.add_argument("--oasst1-lang", type=str, default=None)
    p.add_argument("--oasst1-max-examples", type=int, default=None)

    a = p.parse_args()
    cfg = Config(
        model_id=a.model_id, dataset=a.dataset, method=a.method,
        rank=a.rank, lora_alpha=a.lora_alpha,
        last_k_layers=a.last_k_layers, target_projections=a.target_projections,
        include_mlp=a.include_mlp, manual_targets=a.manual_targets,
        kl_scale=a.kl_scale, kl_freeze_steps=a.kl_freeze_steps, beta_warmup_steps=a.beta_warmup_steps,
        logalpha_thresh=a.logalpha_thresh, prune_every=a.prune_every, min_ranks=a.min_ranks,
        prune_start_step=a.prune_start_step, beta_prune_threshold=a.beta_prune_threshold,
        sample_warmup_steps=a.sample_warmup_steps,
        lr=a.lr, weight_decay=a.weight_decay, max_steps=a.max_steps, warmup_steps=a.warmup_steps,
        batch_size=a.batch_size, grad_accum=a.grad_accum, block_size=a.block_size,
        mc_eval=a.mc_eval, max_eval_tokens=a.max_eval_tokens,
        dtype=a.dtype, seed=a.seed, diag_every=a.diag_every, diag_token_cap=a.diag_token_cap,
        load_in_4bit=a.load_in_4bit,
        arc_split=a.arc_split, arc_answer_format=a.arc_answer_format,
        arc_fewshot=a.arc_fewshot, arc_max_choices=a.arc_max_choices,
        arc_train=a.arc_train, arc_train_mode=a.arc_train_mode,
        oasst1_split=a.oasst1_split, oasst1_val_ratio=a.oasst1_val_ratio,
        oasst1_lang=a.oasst1_lang, oasst1_max_examples=a.oasst1_max_examples,
    )
    return cfg

# ===================== main =====================
def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(cfg)
    if cfg.method == "bayeslora":
        model = add_bayeslora(model, cfg)
    else:
        model = add_lora(model, cfg)

    ds_name = cfg.dataset.lower()

    # ---- OASST1 path ----
    if ds_name in ("oasst1","openassistant","openassistant/oasst1"):
        # Build train/valid
        train_ds = OASST1Dataset(
            tokenizer, split="train", block_size=cfg.block_size,
            lang=cfg.oasst1_lang, max_examples=cfg.oasst1_max_examples,
            seed=cfg.seed, val_ratio=cfg.oasst1_val_ratio,
        )
        valid_ds = OASST1Dataset(
            tokenizer, split="validation", block_size=cfg.block_size,
            lang=cfg.oasst1_lang, max_examples=cfg.oasst1_max_examples,
            seed=cfg.seed, val_ratio=cfg.oasst1_val_ratio,
        )
        pin = torch.cuda.is_available()
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=2 if pin else 0, pin_memory=pin, collate_fn=pad_to_block_collate)
        valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                              num_workers=1 if pin else 0, pin_memory=pin, collate_fn=pad_to_block_collate)

        print(f"[OASST1] Train ex: {len(train_ds)} | Valid ex: {len(valid_ds)} | Block size: {cfg.block_size}")
        if cfg.method == "bayeslora":
            print(f"[BayesLoRA] Trainable params: {count_adapter_params(model):,}")

        model = train_one(cfg, model, tokenizer, train_dl, valid_dl)

        ppl_det, ppl_mc = evaluate_perplexity(cfg, model, valid_dl,
                                              mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
                                              current_step=cfg.max_steps)
        tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
            cfg, model, valid_dl,
            mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
            token_cap=cfg.max_eval_tokens
        )
        print("\n=== Results (OASST1) ===")
        print(f"Method: {cfg.method}")
        print(f"Perplexity (det): {ppl_det:.3f} | Perplexity (MC, {cfg.mc_eval}): {ppl_mc:.3f}")
        print(f"Token-Acc (det): {tokacc_det*100:.2f}% | ECE(det): {ece_det:.4f}")
        print(f"Token-Acc (MC):  {tokacc_mc*100:.2f}% | ECE(MC,{cfg.mc_eval}): {ece_mc:.4f}")
        return

    # ---- ARC path ----
    if ds_name in ("arc-c","arc_challenge","arcchallenge","arc"):
        if cfg.arc_train:
            if cfg.arc_train_mode == "mc":
                train_ds = ARCItemDataset(tokenizer, split="train",  max_choices=cfg.arc_max_choices)
                valid_ds = ARCItemDataset(tokenizer, split="validation", max_choices=cfg.arc_max_choices)
                pin = torch.cuda.is_available()
                train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                      num_workers=2 if pin else 0, pin_memory=pin, collate_fn=arc_item_collate)
                valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                                      num_workers=1 if pin else 0, pin_memory=pin, collate_fn=arc_item_collate)
                print(f"[ARC-MC] Train ex: {len(train_ds)} | Valid ex: {len(valid_ds)}")
                # custom loop
                device = next(model.parameters()).device
                model.train()
                total_steps = cfg.max_steps
                adapter_params = [p for _,p in model.named_parameters() if p.requires_grad]
                opt = torch.optim.AdamW(adapter_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
                sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
                global_step = 0; accum = 0; opt.zero_grad(set_to_none=True)
                import time; last_t = time.time()
                LOG_EVERY = max(1, getattr(cfg,"diag_every",500)//5)
                while global_step < total_steps:
                    for batch_items in train_dl:
                        if global_step >= total_steps: break
                        loss_ce, acc = arc_mc_step(model, tokenizer, batch_items, sample_at_train=(cfg.method=="bayeslora" and global_step>=cfg.sample_warmup_steps))
                        kl_term = torch.tensor(0.0, device=device)
                        beta = 0.0
                        if cfg.method == "bayeslora":
                            if global_step > cfg.kl_freeze_steps:
                                beta = min(1.0, (global_step - cfg.kl_freeze_steps) / max(1, cfg.beta_warmup_steps))
                            kl_term = bayeslora_step(model, beta=beta) / 8.0
                        loss = (loss_ce / cfg.grad_accum) + (kl_term / cfg.grad_accum)
                        loss.backward(); accum += 1
                        if accum >= cfg.grad_accum:
                            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                            accum = 0; global_step += 1
                            if global_step % LOG_EVERY == 0:
                                dt = time.time() - last_t; last_t = time.time()
                                print(f"step {global_step}/{total_steps} | mc_loss {float(loss_ce.item()):.4f} | kl {float(kl_term.item()):.6f} | β {beta:.2f} | acc {float(acc):.3f} | {dt:.2f}s")
                            if (cfg.method == "bayeslora" and cfg.prune_every > 0 and
                                global_step >= (cfg.prune_start_step if cfg.prune_start_step is not None else (cfg.kl_freeze_steps + cfg.beta_warmup_steps)) and
                                beta >= cfg.beta_prune_threshold and
                                global_step % cfg.prune_every == 0):
                                before = count_adapter_params(model)
                                changed = bayeslora_prune(model, logalpha_thresh=cfg.logalpha_thresh, min_ranks=cfg.min_ranks)
                                after = count_adapter_params(model)
                                ranks = get_adapter_ranks(model)
                                print(f"[Prune@{global_step}] adapter params: {before} -> {after}  ranks: {ranks}")
                                if changed:
                                    adapter_params = [p for _,p in model.named_parameters() if p.requires_grad]
                                    opt = torch.optim.AdamW(adapter_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
                                    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
                evaluate_arc(cfg, model, tokenizer, split=cfg.arc_split)
                return
            else:
                train_ds = ARCDataset(tokenizer, split="train",
                                      answer_format=cfg.arc_answer_format,
                                      max_choices=cfg.arc_max_choices,
                                      block_size=cfg.block_size)
                valid_ds = ARCDataset(tokenizer, split="validation",
                                      answer_format=cfg.arc_answer_format,
                                      max_choices=cfg.arc_max_choices,
                                      block_size=cfg.block_size)
                pin = torch.cuda.is_available()
                train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                      num_workers=2 if pin else 0, pin_memory=pin, collate_fn=pad_to_block_collate)
                valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                                      num_workers=1 if pin else 0, pin_memory=pin, collate_fn=pad_to_block_collate)
                print(f"[ARC-LM] Train ex: {len(train_ds)} | Valid ex: {len(valid_ds)} | Block size: {cfg.block_size}")
                if cfg.method == "bayeslora":
                    print(f"[BayesLoRA] Trainable params: {count_adapter_params(model):,}")
                model = train_one(cfg, model, tokenizer, train_dl, valid_dl)
                evaluate_arc(cfg, model, tokenizer, split=cfg.arc_split)
                return
        evaluate_arc(cfg, model, tokenizer, split=cfg.arc_split)
        return

    # ---- LM baselines ----
    if ds_name in ("wikitext2", "wikitext-2", "wikitext-2-raw-v1"):
        train_ds, valid_ds = load_wikitext2(tokenizer, cfg.block_size)
    elif ds_name in ("wikitext103", "wikitext-103", "wikitext-103-raw-v1"):
        train_ds, valid_ds = load_wikitext103(tokenizer, cfg.block_size)
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

    model = train_one(cfg, model, tokenizer, train_dl, valid_dl)

    ppl_det, ppl_mc = evaluate_perplexity(cfg, model, valid_dl,
                                          mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
                                          current_step=cfg.max_steps)
    tokacc_det, ece_det, tokacc_mc, ece_mc = evaluate_token_metrics(
        cfg, model, valid_dl,
        mc_samples=(cfg.mc_eval if cfg.method == "bayeslora" else 1),
        token_cap=cfg.max_eval_tokens
    )
    print("\n=== Results (LM) ===")
    print(f"Method: {cfg.method}")
    print(f"Perplexity (det): {ppl_det:.3f} | Perplexity (MC, {cfg.mc_eval}): {ppl_mc:.3f}")
    print(f"Token-Acc (det): {tokacc_det*100:.2f}% | ECE(det): {ece_det:.4f}")
    print(f"Token-Acc (MC):  {tokacc_mc*100:.2f}% | ECE(MC,{cfg.mc_eval}): {ece_mc:.4f}")

# ---------- ARC-LM dataset class (reused) ----------
class ARCDataset(Dataset):
    def __init__(self, tokenizer, split, answer_format="letter", max_choices=4, block_size=256):
        ds = load_arc_challenge()[split]
        self.pad_id = tokenizer.pad_token_id
        self.block_size = int(block_size)
        self.examples = []
        for ex in ds:
            q = _clean(ex["question"])
            raw_labels = [str(x) for x in ex["choices"]["label"]]
            raw_texts  = [_clean(str(x)) for x in ex["choices"]["text"]]
            ans = _clean(ex["answerKey"])
            labels, texts = _subset_choices(raw_labels, raw_texts, max_choices, ans)
            pairs  = list(zip(labels, texts))
            prompt = build_arc_prompt(q, pairs, fewshots=None)
            if answer_format == "letter":
                target = ans
            else:
                idx = labels.index(ans)
                target = " " + texts[idx]
            enc_p = tokenizer(prompt, add_special_tokens=False)
            enc_t = tokenizer(target, add_special_tokens=False)
            x = enc_p["input_ids"]; t = enc_t["input_ids"]
            ids = x + t; labels_vec = ([-100] * len(x)) + t
            ids = ids[-self.block_size:]; labels_vec = labels_vec[-self.block_size:]
            if len(ids) < self.block_size:
                pad_len = self.block_size - len(ids)
                ids = [self.pad_id] * pad_len + ids
                labels_vec = [-100] * pad_len + labels_vec
            self.examples.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels_vec, dtype=torch.long)))
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

@torch.no_grad()
def evaluate_arc_quick(cfg: Config, model, tokenizer, split="validation", max_items: int = 64, allow_mc: bool = False):
    ds = load_arc_challenge()[split]
    items = []
    for ex in ds:
        items.append(dict(
            question=_clean(ex["question"]),
            choices={"label": ex["choices"]["label"], "text": ex["choices"]["text"]},
            answerKey=_clean(ex["answerKey"])
        ))
        if len(items) >= max_items: break
    batch_prompts, batch_opts = arc_build_batch(
        tokenizer, items,
        answer_format=cfg.arc_answer_format,
        fewshot_k=cfg.arc_fewshot, max_choices=cfg.arc_max_choices,
    )
    mc_samps = (cfg.mc_eval if (cfg.method == "bayeslora" and allow_mc) else 1)
    det, mc = score_arc_batch(model, tokenizer, batch_prompts, batch_opts, mc_samples=mc_samps)
    import numpy as np
    gold = np.array([ord(it["answerKey"][0]) - ord('A') for it in items], dtype=np.int32)
    det_acc = float((det.argmax(-1).cpu().numpy() == gold).mean())
    mc_acc  = float((mc.argmax(-1).cpu().numpy()  == gold).mean())
    return det_acc, mc_acc, len(items)

if __name__ == "__main__":
    main()