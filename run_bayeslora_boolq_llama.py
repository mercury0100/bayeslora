#!/usr/bin/env python3
"""
BayesLoRA on LLaMA-7B: BoolQ fine-tune + eval (det vs MC)
- Formats BoolQ as: Question + Passage -> " True"/" False"
- Trains with BayesLoRA (frozen backbone)
- Evaluates deterministic and MC posterior predictive (antithetic pairs, probability averaging)

Usage (example):
  python run_bayeslora_boolq_llama.py --model_name meta-llama/Llama-2-7b-hf --epochs 1 --mc_T 16
"""

import argparse
import math
import os
import sys
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import local bayeslora.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from bayeslora import (
    BayesLoraConfig, apply_bayeslora,
    bayeslora_regularizers, gate_kl_weight_schedule, temperature_schedule, update_bayeslora_temperatures,
    set_expected_gates, set_sampling_eval, iter_bayeslora_modules
)

# --------------------
# Prompting / formatting
# --------------------
PROMPT_TMPL = (
    "Read the passage and answer the question with 'True' or 'False'.\n\n"
    "Passage: {passage}\n"
    "Question: {question}\n"
    "Answer:"
)

def build_item(tokenizer, question: str, passage: str, answer_bool: bool, max_len: int):
    prompt = PROMPT_TMPL.format(passage=passage.strip(), question=question.strip())
    ans_text = " True" if answer_bool else " False"  # leading space matters for tokenizer
    full = prompt + ans_text

    # Tokenize prompt and full
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)

    # Labels: ignore prompt tokens; supervise only the answer tokens
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Truncate from the left if necessary (keep the answer visible)
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        labels   = labels[overflow:]

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "prompt_len": len(prompt_ids if len(full_ids) <= max_len else prompt_ids)  # best-effort
    }

def collate_pad(batch, pad_id: int):
    # Coerce lists → tensors
    def to_long_tensor(x):
        return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)

    input_ids_list = [to_long_tensor(ex["input_ids"]) for ex in batch]
    labels_list    = [to_long_tensor(ex["labels"])    for ex in batch]

    max_len = max(x.numel() for x in input_ids_list)

    input_ids, labels, prompt_lens = [], [], []
    for ids, lab, ex in zip(input_ids_list, labels_list, batch):
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            # pad labels with -100 so CE ignores padding
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(lab)
        # tolerate missing prompt_len (not used downstream)
        prompt_lens.append(int(ex.get("prompt_len", len(ids) - (lab != -100).sum().item())))

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "prompt_len": torch.tensor(prompt_lens, dtype=torch.long),
    }

# --------------------
# Metrics (binary)
# --------------------
def binary_metrics(p_true: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    p_true: [B] predicted prob of True
    y_true: [B] 0/1 ground-truth (1=True)
    """
    y_pred = (p_true >= 0.5).long()
    acc = (y_pred == y_true).float().mean().item()
    # NLL for true label
    eps = 1e-12
    nll = -(y_true * (p_true+eps).log() + (1 - y_true) * (1 - p_true + eps).log()).mean().item()
    # Brier
    brier = ((p_true - y_true.float()) ** 2).mean().item()
    # ECE
    n_bins = 15
    confidences = torch.where(y_pred.bool(), p_true, 1 - p_true)
    accuracies = (y_pred == y_true).float()
    bin_bounds = torch.linspace(0, 1, n_bins + 1, device=p_true.device)
    ece_val = torch.zeros((), device=p_true.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i+1])
        if in_bin.any():
            ece_val += (in_bin.float().mean() *
                        (confidences[in_bin].mean() - accuracies[in_bin].mean()).abs())
    return {"acc": acc, "nll": nll, "ece": ece_val.item(), "brier": brier}

# --------------------
# Eval helpers (log-likelihood over options)
# --------------------
@torch.no_grad()
def option_logprobs(model, tokenizer, prompts: List[str], options_tok: List[List[int]], device) -> torch.Tensor:
    """
    Compute summed log-probability of each option continuation given each prompt.
    Returns tensor [B, O] of option log-scores.
    """
    assert len(options_tok) >= 2
    B = len(prompts)
    # Pre-tokenize prompts
    prompt_tok = tokenizer(prompts, add_special_tokens=False)
    maxp = max(len(ids) for ids in prompt_tok["input_ids"])
    pad_id = tokenizer.pad_token_id

    def stack_pad(seqs):
        out = []
        for s in seqs:
            t = torch.tensor(s, dtype=torch.long)
            if len(s) < maxp:
                t = torch.cat([t, torch.full((maxp-len(s),), pad_id, dtype=torch.long)])
            out.append(t)
        return torch.stack(out, dim=0)

    prompts_ids = stack_pad(prompt_tok["input_ids"]).to(device)
    attn = (prompts_ids != pad_id).long()

    # For each option, feed prompt+option and accumulate log probs over option tokens
    scores = []
    for opt in options_tok:
        opt_t = torch.tensor(opt, dtype=torch.long, device=device)
        # Build inputs: concat per example
        inputs = []
        labels = []
        for i in range(B):
            pi = prompts_ids[i][attn[i].bool()]  # strip pads
            seq = torch.cat([pi, opt_t], dim=0)
            # Labels: ignore prompt part, supervise option
            lab = torch.cat([torch.full((pi.numel(),), -100, dtype=torch.long, device=device), opt_t], dim=0)
            inputs.append(seq)
            labels.append(lab)
        # Pad to batch
        maxlen = max(x.numel() for x in inputs)
        inp_pad = []
        lab_pad = []
        for i in range(B):
            need = maxlen - inputs[i].numel()
            inp_pad.append(torch.cat([inputs[i], torch.full((need,), pad_id, dtype=torch.long, device=device)]))
            lab_pad.append(torch.cat([labels[i], torch.full((need,), -100, dtype=torch.long, device=device)]))
        inp = torch.stack(inp_pad, dim=0)
        lab = torch.stack(lab_pad, dim=0)
        out = model(input_ids=inp, attention_mask=(inp != pad_id).long(), labels=lab)
        # Sum token-level NLL over option tokens -> log prob
        # out.loss is mean over tokens; we want per-sample sum over option tokens.
        logits = out.logits  # [B, L, V]
        shift_logits = logits[:, :-1, :]
        shift_labels = lab[:, 1:]
        logp = F.log_softmax(shift_logits, dim=-1)
        token_logp = torch.gather(logp, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp.masked_fill(shift_labels.eq(-100), 0.0)
        seq_logp = token_logp.sum(dim=1)  # [B]
        scores.append(seq_logp)
    return torch.stack(scores, dim=1)  # [B, O]

# --------------------
# Antithetic MC over BayesLoRA gates with probability averaging
# --------------------
@torch.no_grad()
def evaluate_mc_boolq(model, tokenizer, ds, device, batch_size: int, T: int = 8, antithetic: bool = True):
    model.eval()
    set_expected_gates(model, False)
    set_sampling_eval(model, True)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)

    # Pre-tokenize answer options once
    true_ids  = tokenizer.encode(" True", add_special_tokens=False)
    false_ids = tokenizer.encode(" False", add_special_tokens=False)
    options = [true_ids, false_ids]

    all_p_true = []
    all_y = []

    modules = list(iter_bayeslora_modules(model))
    def sample_and_set_masks():
        gs = []
        for m in modules:
            g = m._sample_concrete_keep(shape=torch.Size([]), training=False)
            g = (g >= 0.5).to(g.dtype)
            m.set_forced_mask(g)
            gs.append(g)
        return gs
    def set_complement(gs):
        for m, g in zip(modules, gs):
            m.set_forced_mask(1.0 - g)
    def clear_masks():
        for m in modules:
            m.set_forced_mask(None)

    for batch in loader:
        # Batch is a list of dicts: need prompts and labels
        prompts = []
        y_true = []
        for ex in batch:
            # Recover the original strings from cached fields
            prompts.append(PROMPT_TMPL.format(passage=ex["raw_passage"], question=ex["raw_question"]))
            y_true.append(int(ex["raw_answer"]))  # 1=True, 0=False
        y_true = torch.tensor(y_true, device=device)

        # MC probability averaging over options
        T_eff = T
        if antithetic and (T % 2 == 1):
            T_eff = T + 1

        probs_sum = torch.zeros(len(batch), 2, device=device)
        cnt = 0
        while cnt < T_eff:
            if antithetic:
                gs = sample_and_set_masks()
                scores1 = option_logprobs(model, tokenizer, prompts, options, device)  # [B,2]
                p1 = F.softmax(scores1, dim=-1)
                set_complement(gs)
                scores2 = option_logprobs(model, tokenizer, prompts, options, device)
                p2 = F.softmax(scores2, dim=-1)
                clear_masks()
                probs_sum += p1 + p2
                cnt += 2
            else:
                # rely on eval-time sampling
                scores = option_logprobs(model, tokenizer, prompts, options, device)
                probs_sum += F.softmax(scores, dim=-1)
                cnt += 1

        probs = probs_sum / float(T_eff)
        p_true = probs[:, 0]  # index 0 corresponds to " True"
        all_p_true.append(p_true)
        all_y.append(y_true)

    p_true = torch.cat(all_p_true, dim=0)
    y = torch.cat(all_y, dim=0)
    return binary_metrics(p_true, y)

@torch.no_grad()
def evaluate_det_boolq(model, tokenizer, ds, device, batch_size: int):
    model.eval()
    set_expected_gates(model, True)  # deterministic (expected gates)
    set_sampling_eval(model, False)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)
    true_ids  = tokenizer.encode(" True", add_special_tokens=False)
    false_ids = tokenizer.encode(" False", add_special_tokens=False)
    options = [true_ids, false_ids]

    all_p_true = []
    all_y = []
    for batch in loader:
        prompts = []
        y_true = []
        for ex in batch:
            prompts.append(PROMPT_TMPL.format(passage=ex["raw_passage"], question=ex["raw_question"]))
            y_true.append(int(ex["raw_answer"]))
        y_true = torch.tensor(y_true, device=device)

        scores = option_logprobs(model, tokenizer, prompts, options, device)
        probs = F.softmax(scores, dim=-1)
        p_true = probs[:, 0]
        all_p_true.append(p_true)
        all_y.append(y_true)

    p_true = torch.cat(all_p_true, dim=0)
    y = torch.cat(all_y, dim=0)
    return binary_metrics(p_true, y)

# --------------------
# Training loop (causal LM with answer tokens supervised)
# --------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, total_steps, step_offset, cfg: BayesLoraConfig):
    model.train()
    step = step_offset
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        # schedules first
        lambda_gate = gate_kl_weight_schedule(step, total_steps, warmup_frac=cfg.kl_warmup_frac, max_weight=cfg.kl_max_weight)
        tau = temperature_schedule(step, total_steps, start=cfg.tau_init, end=cfg.tau_final, warmup_frac=cfg.tau_warmup_frac)
        update_bayeslora_temperatures(model, tau)

        out = model(input_ids=batch["input_ids"], attention_mask=(batch["input_ids"] != tokenizer.pad_token_id).long(), labels=batch["labels"])
        data_loss = out.loss

        # per-gate normalized KL
        # (we re-use internal helpers then normalize manually)
        from bayeslora import bayeslora_regularizers as _reg, iter_bayeslora_modules as _iter
        l2, kl_raw, _ = _reg(model, lambda_l2=cfg.lambda_l2, lambda_gate=1.0, prior_keep=cfg.prior_keep)
        ngates = sum(m.alpha.numel() for m in _iter(model))
        kl_per_gate = kl_raw / max(1, ngates)
        reg = cfg.lambda_l2 * l2 + lambda_gate * kl_per_gate

        loss = data_loss + reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i % 50 == 0:
            print(f"step {step} | loss={loss.item():.4f} | data={data_loss.item():.4f} | l2={l2.item():.4f} "
                  f"| kl_raw={kl_raw.item():.4f} | kl_pg={kl_per_gate.item():.6f} | tau={tau:.3f} | lam_gate={lambda_gate:.3f}")
        step += 1
    return step

def logit(x: float, eps: float = 1e-6) -> float:
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x) - math.log(1 - x)

def init_gates_to_prior(model, prior_keep: float):
    prior_drop = 1.0 - float(prior_keep)
    with torch.no_grad():
        for m in iter_bayeslora_modules(model):
            m.alpha.fill_(logit(prior_drop))

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--prior_keep", type=float, default=0.7)
    parser.add_argument("--tau_init", type=float, default=2.0)
    parser.add_argument("--tau_final", type=float, default=0.5)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.3)
    parser.add_argument("--tau_warmup_frac", type=float, default=0.3)
    parser.add_argument("--kl_max_weight", type=float, default=0.05)
    parser.add_argument("--lambda_l2", type=float, default=1e-4)
    parser.add_argument("--mc_T", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=0)  # 0 = full
    parser.add_argument("--limit_val", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Apply BayesLoRA BEFORE .to(device)
    cfg = BayesLoraConfig(
        r=args.r, lora_alpha=args.lora_alpha,
        prior_keep=args.prior_keep,
        tau_init=args.tau_init, tau_final=args.tau_final,
        kl_warmup_frac=args.kl_warmup_frac, tau_warmup_frac=args.tau_warmup_frac,
        kl_max_weight=args.kl_max_weight, lambda_l2=args.lambda_l2,
        target_modules=("q_proj","k_proj","v_proj","o_proj","up_proj"),#,"down_proj","gate_proj"),
        freeze_base=True,
    )
    apply_bayeslora(model, cfg, verbose=True)
    init_gates_to_prior(model, cfg.prior_keep)
    model.to(device)

    # ----- Data: BoolQ -----
    raw = load_dataset("boolq")
    train_raw = raw["train"]
    val_raw = raw["validation"]  # BoolQ has "validation" split; treat as val

    def to_item(ex):
        item = build_item(tokenizer, ex["question"], ex["passage"], bool(ex["answer"]), max_len=args.max_len)
        # Keep originals for eval prompts
        item["raw_question"] = ex["question"]
        item["raw_passage"] = ex["passage"]
        item["raw_answer"] = bool(ex["answer"])
        return item

    train_ds = train_raw.map(to_item).remove_columns([c for c in train_raw.column_names])
    val_ds   = val_raw.map(to_item).remove_columns([c for c in val_raw.column_names])

    if args.limit_train > 0:
        train_ds = train_ds.select(range(args.limit_train))
    if args.limit_val > 0:
        val_ds = val_ds.select(range(args.limit_val))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id)
    )
    # For eval we need raw strings, so we pass list of dicts (no collator)
    val_list = [{"raw_question": x["raw_question"], "raw_passage": x["raw_passage"], "raw_answer": x["raw_answer"]} for x in val_ds]

    # ----- Optim & sched -----
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # ----- Train -----
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        iter_loader = iter(train_loader)
        for i in range(len(train_loader)):
            batch = next(iter_loader)
            for k in batch:
                batch[k] = batch[k].to(device)

            # schedules
            lambda_gate = gate_kl_weight_schedule(step, total_steps, warmup_frac=cfg.kl_warmup_frac, max_weight=cfg.kl_max_weight)
            tau = temperature_schedule(step, total_steps, start=cfg.tau_init, end=cfg.tau_final, warmup_frac=cfg.tau_warmup_frac)
            update_bayeslora_temperatures(model, tau)

            out = model(input_ids=batch["input_ids"], attention_mask=(batch["input_ids"] != tokenizer.pad_token_id).long(), labels=batch["labels"])
            data_loss = out.loss

            from bayeslora import bayeslora_regularizers as _reg, iter_bayeslora_modules as _iter
            l2, kl_raw, _ = _reg(model, lambda_l2=cfg.lambda_l2, lambda_gate=1.0, prior_keep=cfg.prior_keep)
            ngates = sum(m.alpha.numel() for m in _iter(model))
            kl_per_gate = kl_raw / max(1, ngates)
            reg = cfg.lambda_l2 * l2 + lambda_gate * kl_per_gate

            loss = (data_loss + reg) / max(1, args.grad_accum)
            loss.backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if i % (10 * max(1, args.grad_accum)) == 0:
                print(f"epoch {epoch+1} step {step} | data={data_loss.item():.4f} | l2={l2.item():.4f} "
                      f"| kl_raw={kl_raw.item():.4f} | kl_pg={kl_per_gate.item():.6f} | tau={tau:.3f} | lam_gate={lambda_gate:.3f}")

            step += 1

        # ----- Eval each epoch -----
        det = evaluate_det_boolq(model, tokenizer, val_list, device, batch_size=8)
        mc  = evaluate_mc_boolq(model, tokenizer, val_list, device, batch_size=4, T=args.mc_T, antithetic=True)
        print(f"\n[Epoch {epoch+1}] BoolQ Validation")
        print("  Deterministic:", det)
        print(f"  MC (T={args.mc_T}, antithetic):", mc)

    # Final
    det = evaluate_det_boolq(model, tokenizer, val_list, device, batch_size=8)
    mc  = evaluate_mc_boolq(model, tokenizer, val_list, device, batch_size=4, T=args.mc_T, antithetic=True)
    print("\n=== Final BoolQ Validation ===")
    print("Deterministic:", det)
    print(f"MC (T={args.mc_T}, antithetic):", mc)

if __name__ == "__main__":
    main()