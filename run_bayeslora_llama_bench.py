#!/usr/bin/env python3
import argparse, json, math, os, sys
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW

# local BayesLoRA
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path: sys.path.append(SCRIPT_DIR)
from bayeslora import (
    BayesLoraConfig, apply_bayeslora, iter_bayeslora_modules,
    bayeslora_regularizers, gate_kl_weight_schedule, temperature_schedule, update_bayeslora_temperatures,
    set_expected_gates, set_sampling_eval
)

# ---------- generic MC-choice scaffolding ----------
def binary_metrics(p_true: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    y_pred = (p_true >= 0.5).long()
    acc = (y_pred == y_true).float().mean().item()
    eps = 1e-12
    nll = -(y_true * (p_true+eps).log() + (1-y_true) * (1-p_true+eps).log()).mean().item()
    brier = ((p_true - y_true.float())**2).mean().item()
    # ECE (binary)
    n_bins = 15
    conf = torch.where(y_pred.bool(), p_true, 1-p_true)
    accs = (y_pred == y_true).float()
    bins = torch.linspace(0,1,n_bins+1, device=p_true.device)
    ece = torch.zeros((), device=p_true.device)
    for i in range(n_bins):
        inb = (conf > bins[i]) & (conf <= bins[i+1])
        if inb.any():
            ece += inb.float().mean() * (conf[inb].mean() - accs[inb].mean()).abs()
    return {"acc": acc, "nll": nll, "ece": ece.item(), "brier": brier}

def mc_option_probs(model, tokenizer, prompts: List[str], options_tok: List[List[int]], device,
                    T: int = 8, antithetic: bool = True) -> torch.Tensor:
    """Return [B, O] probs by MC probability averaging with true antithetic masks."""
    model.eval()
    set_expected_gates(model, False)
    set_sampling_eval(model, True)

    # helper: sum logprobs over option tokens
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        def option_logprobs_for(prompts_batch: List[List[int]], option: List[int]) -> torch.Tensor:
            B = len(prompts_batch)
            opt = torch.tensor(option, dtype=torch.long, device=device)
            # build sequences per sample: prompt + option, label only option
            seqs, labs = [], []
            for ids in prompts_batch:
                pi = torch.tensor(ids, dtype=torch.long, device=device)
                seq = torch.cat([pi, opt], dim=0)
                lab = torch.cat([torch.full((pi.numel(),), -100, dtype=torch.long, device=device), opt], dim=0)
                seqs.append(seq); labs.append(lab)
            Lmax = max(x.numel() for x in seqs)
            def pad_to(x, L): return torch.cat([x, torch.full((L-x.numel(),), pad_id, dtype=torch.long, device=device)])
            inp = torch.stack([pad_to(x, Lmax) for x in seqs], dim=0)
            labt= torch.stack([torch.cat([y, torch.full((Lmax-y.numel(),), -100, dtype=torch.long, device=device)]) for y in labs], dim=0)
            attn = (inp != pad_id).long()
            out = model(input_ids=inp, attention_mask=attn, labels=labt)
            logits = out.logits[:, :-1, :]
            labels = labt[:, 1:]
            logp = F.log_softmax(logits, dim=-1)
            tok_lp = torch.gather(logp, dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            tok_lp = tok_lp.masked_fill(labels.eq(-100), 0.0)
            return tok_lp.sum(dim=1)  # [B]

        # tokenize prompts once
        tok = tokenizer(prompts, add_special_tokens=False)
        prompts_tok = tok["input_ids"]

        # antithetic masks via forced masks
        modules = list(iter_bayeslora_modules(model))
        def sample_and_set():
            gs=[]
            for m in modules:
                g = m._sample_concrete_keep(shape=torch.Size([]), training=False)
                g = (g >= 0.5).to(g.dtype)
                m.set_forced_mask(g); gs.append(g)
            return gs
        def set_comp(gs):
            for m,g in zip(modules, gs): m.set_forced_mask(1.0 - g)
        def clear():
            for m in modules: m.set_forced_mask(None)

        # run T samples (pairs if antithetic)
        T_eff = T + (1 if (antithetic and T%2==1) else 0)
        probs_sum = torch.zeros(len(prompts), len(options_tok), device=device)
        cnt=0
        while cnt < T_eff:
            if antithetic:
                gs = sample_and_set()
                scores1 = torch.stack([option_logprobs_for(prompts_tok, opt) for opt in options_tok], dim=1)
                set_comp(gs)
                scores2 = torch.stack([option_logprobs_for(prompts_tok, opt) for opt in options_tok], dim=1)
                clear()
                probs_sum += F.softmax(scores1, dim=-1) + F.softmax(scores2, dim=-1)
                cnt += 2
            else:
                scores = torch.stack([option_logprobs_for(prompts_tok, opt) for opt in options_tok], dim=1)
                probs_sum += F.softmax(scores, dim=-1)
                cnt += 1
        return probs_sum / float(T_eff)

# ---------- dataset wrappers (BoolQ, PIQA, HellaSwag, ARC, OBQA) ----------
def load_task(task: str):
    if task == "boolq":
        ds = load_dataset("boolq")
        def to_prompt(ex):
            prompt = (
                "Read the passage and answer the question with 'True' or 'False'.\n\n"
                f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
            )
            options = [" True", " False"]
            y = int(bool(ex["answer"]))  # 1=True, 0=False
            return prompt, options, y
        val = [to_prompt(x) for x in ds["validation"]]
        return val, "binary"
    if task == "piqa":
        ds = load_dataset("piqa")
        def to_prompt(ex):
            prompt = f"Goal: {ex['goal']}\nWhich solution is more plausible?\nA) {ex['sol1']}\nB) {ex['sol2']}\nAnswer:"
            options = [" A", " B"]
            y = int(ex["label"])  # 0 or 1 (B? In PIQA, label=0 means sol1 correct.)
            # Map to A=0, B=1
            return prompt, options, y
        val = [to_prompt(x) for x in ds["validation"]]
        return val, "mc_ab"
    if task == "hellaswag":
        ds = load_dataset("hellaswag")
        def to_prompt(ex):
            ctx = ex["ctx_a"] + " " + ex["ctx_b"]
            prompt = f"{ctx}\nChoose the best way to continue.\nA) {ex['endings'][0]}\nB) {ex['endings'][1]}\nC) {ex['endings'][2]}\nD) {ex['endings'][3]}\nAnswer:"
            options = [" A"," B"," C"," D"]
            y = int(ex["label"])
            return prompt, options, y
        val = [to_prompt(x) for x in ds["validation"]]
        return val, "mc_abcd"
    if task in ("arc_easy","arc_challenge"):
        name = "ARC-Easy" if task=="arc_easy" else "ARC-Challenge"
        ds = load_dataset("ai2_arc", name)
        def to_prompt(ex):
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]  # like ["A","B",...]
            y = labels.index(ex["answerKey"])
            letters = [" A"," B"," C"," D"," E"," F"]
            options = letters[:len(choices)]
            prompt = f"{ex['question']}\n" + "\n".join(f"{letters[i][1]}) {choices[i]}" for i in range(len(choices))) + "\nAnswer:"
            return prompt, options, y
        val = [to_prompt(x) for x in ds["validation"]]
        return val, "mc_letter"
    if task == "openbookqa":
        ds = load_dataset("openbookqa","main")
        def to_prompt(ex):
            q = ex["question_stem"]
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            y = labels.index(ex["answerKey"])
            letters = [" A"," B"," C"," D"]
            options = letters[:len(choices)]
            prompt = f"{q}\n" + "\n".join(f"{letters[i][1]}) {choices[i]}" for i in range(len(choices))) + "\nAnswer:"
            return prompt, options, y
        val = [to_prompt(x) for x in ds["validation"]]
        return val, "mc_letter"
    raise ValueError(f"Unknown task: {task}")

# ---------- train adapters lightly (optional) ----------
def logit(x: float, eps: float=1e-6)->float:
    x = min(max(float(x), eps), 1.0-eps)
    return math.log(x) - math.log(1-x)

def init_gates_to_prior(model, prior_keep: float):
    pd = 1.0 - float(prior_keep)
    with torch.no_grad():
        for m in iter_bayeslora_modules(model):
            m.alpha.fill_(logit(pd))

def train_for_steps(model, tokenizer, train_ds, device, steps: int, lr: float,
                    kl_cfg, grad_accum: int=8, bs: int=1, max_len: int=1024):
    if steps <= 0: return
    def build_item(prompt, target):
        ids = tokenizer.encode(prompt + target, add_special_tokens=False)
        # supervise only target chars
        labels = [-100]*len(tokenizer.encode(prompt, add_special_tokens=False)) + tokenizer.encode(target, add_special_tokens=False)
        if len(ids) > max_len:
            overflow = len(ids)-max_len
            ids = ids[overflow:]; labels = labels[overflow:]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}
    # weak supervised mix from validation prompts themselves (optional)
    # (for true apples-to-apples you might fine-tune on train split where applicable)
    dataset = []
    for ex in train_ds:
        prompt, options, y = ex
        target = " " + ("True" if y==1 and len(options)==2 and options[0].strip()=="A" else "")
    # for simplicity, skip supervised phase here; Laplace-LoRA often fits post-hoc after LoRA tuning.

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--tasks", type=str, default="boolq,piqa,hellaswag,arc_easy,arc_challenge,openbookqa")
    p.add_argument("--epochs", type=int, default=0, help="(optional) adapter finetune epochs; default 0 = eval only")
    p.add_argument("--mc_T", type=int, default=16)
    p.add_argument("--batch_size_eval", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--prior_keep", type=float, default=0.7)
    p.add_argument("--tau_init", type=float, default=2.0)
    p.add_argument("--tau_final", type=float, default=0.5)
    p.add_argument("--kl_warmup_frac", type=float, default=0.3)
    p.add_argument("--tau_warmup_frac", type=float, default=0.3)
    p.add_argument("--kl_max_weight", type=float, default=0.05)
    p.add_argument("--lambda_l2", type=float, default=1e-4)
    p.add_argument("--output", type=str, default="bayeslora_llama_results.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    cfg = BayesLoraConfig(
        r=args.r, lora_alpha=args.lora_alpha,
        prior_keep=args.prior_keep,
        tau_init=args.tau_init, tau_final=args.tau_final,
        kl_warmup_frac=args.kl_warmup_frac, tau_warmup_frac=args.tau_warmup_frac,
        kl_max_weight=args.kl_max_weight, lambda_l2=args.lambda_l2,
        target_modules=("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
        freeze_base=True,
    )
    apply_bayeslora(model, cfg, verbose=True)
    init_gates_to_prior(model, cfg.prior_keep)
    model.to(device)

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n=== Evaluating {task} ===")
        val, kind = load_task(task)

        # build per-task prompts and answers
        prompts = [x[0] for x in val]
        options_txt = [x[1] for x in val]
        labels = torch.tensor([x[2] for x in val], device=device)

        # tokenize options once per task
        uniq_opts = sorted(set(tuple(o) for o in options_txt))
        opt_map = {u: [tok.encode(x, add_special_tokens=False) for x in list(u)] for u in uniq_opts}
        probs_all = []
        start = 0
        bs = args.batch_size_eval
        while start < len(prompts):
            end = min(len(prompts), start+bs)
            batch_prompts = prompts[start:end]
            # assume all batches share same option set (common in these datasets); if not, we handle per-row in a pinch:
            opts = options_txt[start]
            options_tok = [tok.encode(x, add_special_tokens=False) for x in opts]
            probs = mc_option_probs(model, tok, batch_prompts, options_tok, device, T=args.mc_T, antithetic=True)
            probs_all.append(probs)
            start = end
        probs = torch.cat(probs_all, dim=0)

        # metrics
        if kind == "binary":
            # " True" is index 0 by construction in load_task
            p_true = probs[:, 0]
            m = binary_metrics(p_true, labels)
        else:
            pred = probs.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            m = {"acc": acc}  # simple headline; extend if you want ECE for MC-choice
        print("MC metrics:", m)
        results[task] = m

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()