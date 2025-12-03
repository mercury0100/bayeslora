#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeBERTa GLUE: BayesLoRA rank dynamics + pruning schedule + head-to-head vs AdaLoRA.

Example:
python run_glue_bench.py \
  --tasks sst2 qnli cola qqp \
  --seeds 1 2 3 \
  --kl_scale 1e-6 \
  --log_alpha_threshold 4.0 \
  --max_steps 9999 \
  --eval_steps 200 \
  --prune_every 2000

python run_glue_bench.py \
  --tasks mnli rte  \
  --seeds 1 2 3 \
  --kl_scale 1e-6 \
  --log_alpha_threshold 4.0 \
  --max_steps 9999 \
  --eval_steps 200 \
  --prune_every 2000
"""

import os, math, argparse, numpy as np, torch, pandas as pd
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from typing import Dict, List, Any

from glue_helpers import (
    set_seed, GLUE_NUM_LABELS, default_targets_for_deberta, build_datasets,
    plot_rank_dynamics, plot_accuracy_over_steps, plot_accuracy_comparison,
    TrainConfig, create_model, evaluate_metrics,
    train_bayeslora, train_adalora, aggregate_rank_logs
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--tasks", type=str, nargs="+", default=["sst2","mnli","qnli","rte","cola","mrpc","qqp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    # BayesLoRA
    parser.add_argument("--kl_scale", type=float, default=1e-6)
    parser.add_argument("--prune_every", type=int, default=2000)
    parser.add_argument("--prune_start", type=int, default=None)
    parser.add_argument("--log_alpha_threshold", type=float, default=3.0)
    parser.add_argument("--lora_r_init", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_DATASETS_CACHE", None))
    args = parser.parse_args()

    # Environment probe
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}", flush=True)

    timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir)
    results_csv = out_root / "results" / f"{timestamp}_glue_bayeslora_adalora.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    th = float(args.log_alpha_threshold)
    print(f"\n=== BayesLoRA / AdaLoRA | log_alpha_threshold={th:.2f} ===", flush=True)

    for task in args.tasks:
        print(f"\n=== Task: {task} ===", flush=True)
        num_labels = GLUE_NUM_LABELS[task]
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        train, val = build_datasets(task, cache_dir=args.cache_dir)
        target_modules = default_targets_for_deberta()
        num_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
        steps_per_epoch = math.ceil(len(train) / args.batch_size)

        # ---------- BayesLoRA ----------
        bayes_acc_logs_by_seed, bayes_rows, end_ranks_per_seed = {}, [], []
        for seed in args.seeds:
            set_seed(seed)
            run_dir = out_root / "checkpoints" / task / f"bayeslora_th{th:.2f}_seed{seed}"
            cfg = TrainConfig(
                model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                kl_scale=args.kl_scale, prune_every=args.prune_every, prune_start=args.prune_start,
                log_alpha_threshold=th, lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                seeds=tuple(args.seeds), out_dir=args.out_dir, target_modules=target_modules
            )

            model = create_model(num_labels=num_labels, model_name=cfg.model_name)
            out = train_bayeslora(cfg, train, val, model, tokenizer, target_modules, seed, run_dir)
            metrics = evaluate_metrics(out["model"], val, tokenizer, task, args.max_len, device=torch.device(args.device))

            bayes_acc_logs_by_seed[seed] = out["acc_log"]
            end_ranks_per_seed.append(out["budget_total_rank"])
            bayes_rows.append({
                "task": task, "model": args.model, "method": "BayesLoRA",
                "log_alpha_thresh": th, "final_total_rank": out["budget_total_rank"],
                "rank_budget_used": out["budget_total_rank"], "accuracy": metrics["accuracy"],
                "nll": metrics["nll"], "ece": metrics["ece"],
                "params_M": out["adapter_params"]/1e6, "train_time_min": out["runtime_min"],
                "seed": seed, "notes": f"prune_every={args.prune_every}"
            })

        # Rank plots
        plot_accuracy_over_steps(task, bayes_acc_logs_by_seed,
                                 out_root/"plots"/f"{task}_th{th:.2f}_bayeslora_acc.png",
                                 title_suffix=f"th={th:.2f}")
        logs_all = aggregate_rank_logs([
            out_root/"checkpoints"/task/f"bayeslora_th{th:.2f}_seed{seed}"/"bayeslora_ranks_log.json"
            for seed in args.seeds
        ])
        if logs_all:
            plot_rank_dynamics(task, logs_all,
                               out_root/"plots"/f"{task}_th{th:.2f}_bayeslora_ranks.png",
                               num_layers=num_layers)

        # ---------- Budgets ----------
        if not end_ranks_per_seed:
            print("[warn] No BayesLoRA ranks; skipping.", flush=True)
            continue
        median_budget = int(np.median(end_ranks_per_seed))
        print(f"[{task}] BayesLoRA per-seed budgets: {end_ranks_per_seed} | median={median_budget}", flush=True)

        # ---------- AdaLoRA ----------
        adalora_rows, adalora_acc_logs_by_seed = [], {}
        for i, seed in enumerate(args.seeds):
            budget = end_ranks_per_seed[i]
            set_seed(seed)
            run_dir = out_root/"checkpoints"/task/f"adalora_th{th:.2f}_seed{seed}"
            base_model = create_model(num_labels=num_labels, model_name=args.model)
            out = train_adalora(
                cfg=TrainConfig(
                    model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                    epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                    adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                    lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                    seeds=tuple(args.seeds), out_dir=args.out_dir, target_modules=target_modules
                ),
                train=train, val=val, model=base_model, tokenizer=tokenizer,
                target_modules=target_modules, seed=seed, run_dir=run_dir,
                budget_total_rank=budget
            )
            metrics = evaluate_metrics(out["model"], val, tokenizer, task, args.max_len, device=torch.device(args.device))
            adalora_acc_logs_by_seed[seed] = out["acc_log"]
            adalora_rows.append({
                "task": task, "model": args.model, "method": "AdaLoRA",
                "final_total_rank": out["budget_total_rank"], "rank_budget_used": budget,
                "accuracy": metrics["accuracy"], "nll": metrics["nll"], "ece": metrics["ece"],
                "params_M": out["adapter_params"]/1e6, "train_time_min": out["runtime_min"],
                "seed": seed, "notes": f"budget={budget} from BayesLoRA seed{seed}"
            })

        plot_accuracy_over_steps(task, adalora_acc_logs_by_seed,
                                 out_root/"plots"/f"{task}_th{th:.2f}_adalora_acc.png",
                                 title_suffix=f"th={th:.2f}")
        logs_all = aggregate_rank_logs([
            out_root/"checkpoints"/task/f"adalora_th{th:.2f}_seed{seed}"/"adalora_ranks_log.json"
            for seed in args.seeds
        ])
        if logs_all:
            plot_rank_dynamics(task, logs_all,
                               out_root/"plots"/f"{task}_th{th:.2f}_adalora_ranks.png",
                               num_layers=num_layers)

        # ---------- Comparison + CSV ----------
        plot_accuracy_comparison(
            task, bayes=bayes_acc_logs_by_seed,
            adalora=adalora_acc_logs_by_seed, lora=None,
            out_path=out_root/"plots"/f"{task}_th{th:.2f}_comparison.png",
            title_suffix=f"th={th:.2f}"
        )

        df_task = pd.DataFrame(bayes_rows + adalora_rows)
        if results_csv.exists():
            df_prev = pd.read_csv(results_csv)
            df_task = pd.concat([df_prev, df_task], ignore_index=True)
        df_task.to_csv(results_csv, index=False)
        print(f"[{task}] Done. Results → {results_csv}", flush=True)

if __name__ == "__main__":
    main()