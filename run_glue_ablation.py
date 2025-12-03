#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLUE ablation runner:
- For each dataset, sweep BayesLoRA log_alpha_thresholds across seeds.
- For each threshold, run AdaLoRA once (single seed) with:
    * init_r matched to BayesLoRA's lora_r_init
    * final rank budget matched to BayesLoRA's median final_total_rank across seeds
- Produce per-dataset results CSV and two figures:
    (A) Mean BayesLoRA rank dynamics (±1 std) and mean accuracy (±1 std), per-threshold curves
    (B) Final accuracy vs log_alpha_threshold: BayesLoRA (±1 std) vs AdaLoRA

Example: python run_glue_ablation.py \
    --tasks sst2 qnli qq\
    --seeds 1 2 3 \
    --kl_scale 1e-6 \
    --log_alpha_thresholds 2.0 3.0 4.0 5.0 \
    --epochs 3
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_JIT", "0")

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoConfig, AutoTokenizer

# Unified helpers (standardized trainers + IO/plot helpers)
from glue_helpers import (
    set_seed,
    GLUE_NUM_LABELS,
    default_targets_for_deberta,
    build_datasets,
    TrainConfig,
    create_model,
    train_bayeslora,
    train_adalora,
)

# ------------------------------------------------------------
# Plotting utilities (local to this script)
# ------------------------------------------------------------
def _mean_std_on_grid(logs_by_seed: dict, value_key: str, grid_steps: np.ndarray):
    """Forward-fill per seed onto a common step grid; return mean, std."""
    if not logs_by_seed:
        return np.full_like(grid_steps, np.nan, float), np.full_like(grid_steps, np.nan, float)

    rows = []
    for _, logs in logs_by_seed.items():
        d = {int(e["step"]): float(e.get(value_key, np.nan)) for e in logs if value_key in e}
        arr = []
        last = np.nan
        for s in grid_steps:
            if int(s) in d and d[int(s)] == d[int(s)]:  # not NaN
                last = d[int(s)]
            arr.append(last)
        arr = np.array(arr, dtype=float)
        # fill leading NaNs with first non-NaN (if any)
        if np.any(~np.isnan(arr)):
            first = next((v for v in arr if not np.isnan(v)), np.nan)
            for i in range(len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = first
                else:
                    break
            for i in range(1, len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = arr[i - 1]
        rows.append(arr)
    M = np.vstack(rows)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0)


def plot_bayes_mean_dynamics(
    task: str,
    out_dir: Path,
    thresholds: list,
    per_threshold_bayes_rank_logs_by_seed: dict,
    per_threshold_bayes_acc_logs_by_seed: dict,
):
    """
    One figure, two panels.
    Top: mean±std total rank over steps, one curve per log_alpha_threshold
    Bottom: mean±std accuracy over steps, one curve per log_alpha_threshold
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of steps across all thresholds/seeds
    all_steps = set()
    for th in thresholds:
        for logs in per_threshold_bayes_rank_logs_by_seed.get(th, {}).values():
            for e in logs:
                all_steps.add(int(e["step"]))
        for logs in per_threshold_bayes_acc_logs_by_seed.get(th, {}).values():
            for e in logs:
                all_steps.add(int(e["step"]))
    if not all_steps:
        return
    grid = np.array(sorted(all_steps), dtype=int)
    xmax = float(grid[-1])

    fig = plt.figure(figsize=(11.8, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4"])

    # Panel 1: total rank dynamics
    for i, th in enumerate(sorted(thresholds)):
        # normalize rank logs to have total_rank explicitly
        normed = {}
        for seed, logs in per_threshold_bayes_rank_logs_by_seed.get(th, {}).items():
            l2 = []
            for e in logs:
                tr = e.get("total_rank")
                if tr is None:
                    pm = e.get("per_module", {})
                    tr = int(sum(int(v) for v in pm.values()))
                l2.append({"step": int(e["step"]), "total_rank": float(tr)})
            normed[seed] = l2
        mean_r, std_r = _mean_std_on_grid(normed, "total_rank", grid)
        ax1.plot(grid, mean_r, linewidth=2, color=colors[i % len(colors)], label=f"th={th:.2f}")
        ax1.fill_between(grid, mean_r - std_r, mean_r + std_r, alpha=0.12, color=colors[i % len(colors)])
    ax1.set_title(f"{task.upper()} – BayesLoRA rank dynamics (mean ± std) across thresholds")
    ax1.set_ylabel("Total rank")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(ncol=2, fontsize=9)

    # Panel 2: accuracy dynamics
    for i, th in enumerate(sorted(thresholds)):
        mean_a, std_a = _mean_std_on_grid(per_threshold_bayes_acc_logs_by_seed.get(th, {}), "accuracy", grid)
        ax2.plot(grid, mean_a, linewidth=2, color=colors[i % len(colors)], label=f"th={th:.2f}")
        ax2.fill_between(grid, mean_a - std_a, mean_a + std_a, alpha=0.12, color=colors[i % len(colors)])
    ax2.set_title("Accuracy over steps (mean ± std)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_bayes_mean_dynamics_all_thresholds.png", dpi=200)
    plt.close(fig)


def plot_final_acc_comparison(task, out_dir: Path, thresholds, bayes_results, adalora_results):
    """
    thresholds: iterable of float
    bayes_results: dict[th] -> list[acc_across_seeds]
    adalora_results: dict[th] -> float (single-seed run)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ths = sorted(thresholds)
    x = np.array(ths, dtype=float)

    bayes_mean = []
    bayes_std  = []
    adalora_y  = []

    for th in ths:
        bvals = bayes_results.get(th, [])
        bayes_mean.append(float(np.mean(bvals)) if bvals else np.nan)
        bayes_std .append(float(np.std (bvals)) if bvals else np.nan)
        aval = adalora_results.get(th, np.nan)
        adalora_y.append(float(aval) if aval == aval else np.nan)

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(1,1,1)

    ax.errorbar(x, bayes_mean, yerr=bayes_std, fmt="-o", capsize=4, label="BayesLoRA", linewidth=2, markersize=5)
    ax.plot(x, adalora_y, "-s", label="AdaLoRA", linewidth=2, markersize=5)

    ax.set_xlabel("log_alpha_threshold")
    ax.set_ylabel("Final accuracy")
    ax.set_title(f"{task.upper()} – Final accuracy vs log_alpha_threshold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in x])

    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_final_acc_adalora_vs_bayes.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--tasks", type=str, nargs="+", default=["sst2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", type=str, default="cuda:0")

    # training core
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)

    # optim
    ap.add_argument("--adapter_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    # bayeslora specifics
    ap.add_argument("--kl_scale", type=float, default=1e-6)
    ap.add_argument("--prune_every", type=int, default=1000)
    ap.add_argument("--prune_start", type=int, default=None)
    ap.add_argument("--lora_r_init", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--log_alpha_thresholds", type=float, nargs="+", default=[3.0])

    # io
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--cache_dir", type=str, default=os.environ.get("HF_DATASETS_CACHE", None))
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    results_dir = out_root / "results"
    plots_dir = out_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # CUDA probe (optional)
    try:
        print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[env] device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}", flush=True)
    except Exception as e:
        print(f"[env] torch/cuda probe failed: {e}", flush=True)

    # Early CUDA touch
    if args.device.startswith("cuda") and torch.cuda.is_available():
        _ = torch.randn(1, device=args.device)
        torch.cuda.synchronize()

    for task in args.tasks:
        print(f"\n===== DATASET: {task} =====", flush=True)
        num_labels = GLUE_NUM_LABELS[task]
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        train, val = build_datasets(task, cache_dir=args.cache_dir)
        target_modules = default_targets_for_deberta()
        _ = AutoConfig.from_pretrained(args.model).num_hidden_layers  # may be handy for future plotting

        per_threshold_final_budget = {}
        per_threshold_final_bayes_accs = {}
        per_threshold_bayes_rank_logs_by_seed = {}  # for figure A (ranks)
        per_threshold_bayes_acc_logs_by_seed  = {}  # for figure A (accuracy)
        adalora_final_accs = {}

        for th in args.log_alpha_thresholds:
            print(f"\n--- BayesLoRA: threshold {th:.2f} ---", flush=True)
            bayes_acc_logs_by_seed = {}
            bayes_rank_logs_by_seed = {}
            bayes_final_ranks = []
            bayes_final_accs = []

            # multi-seed BayesLoRA
            for seed in args.seeds:
                set_seed(seed)
                run_dir = out_root / "checkpoints" / f"{task}" / f"bayeslora_th{th:.2f}_seed{seed}"
                cfg = TrainConfig(
                    model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                    epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                    adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                    kl_scale=args.kl_scale, prune_every=args.prune_every, prune_start=args.prune_start,
                    log_alpha_threshold=th, lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                    out_dir=args.out_dir, target_modules=target_modules, trainable_allowlist=("classifier",),
                )
                model = create_model(num_labels=num_labels, model_name=cfg.model_name)
                out = train_bayeslora(cfg, train, val, model, tokenizer, target_modules, seed, run_dir)

                bayes_acc_logs_by_seed[seed]  = out["acc_log"]
                bayes_rank_logs_by_seed[seed] = out["ranks_log"]
                bayes_final_ranks.append(out["budget_total_rank"])
                bayes_final_accs.append(out["best_acc"])

            per_threshold_bayes_rank_logs_by_seed[th] = bayes_rank_logs_by_seed
            per_threshold_bayes_acc_logs_by_seed[th]  = bayes_acc_logs_by_seed

            final_budget = int(np.median(bayes_final_ranks)) if bayes_final_ranks else 0
            per_threshold_final_budget[th] = final_budget
            per_threshold_final_bayes_accs[th] = list(bayes_final_accs)
            print(f"[{task} th={th:.2f}] median BayesLoRA final rank budget: {final_budget}", flush=True)

            # single-seed AdaLoRA at matched budget
            if final_budget > 0:
                seed = int(args.seeds[0])
                set_seed(seed)
                ada_model = create_model(num_labels=num_labels, model_name=args.model)
                ada_out = train_adalora(
                    cfg=TrainConfig(
                        model_name=args.model, task=task, max_len=args.max_len, batch_size=args.batch_size,
                        epochs=args.epochs, max_steps=args.max_steps, eval_steps=args.eval_steps, device=args.device,
                        adapter_lr=args.adapter_lr, head_lr=args.head_lr, wd=args.wd, warmup_ratio=args.warmup_ratio,
                        lora_r_init=args.lora_r_init, lora_alpha=args.lora_alpha,
                        out_dir=args.out_dir, target_modules=target_modules, trainable_allowlist=("classifier",),
                    ),
                    train=train, val=val, model=ada_model, tokenizer=tokenizer,
                    target_modules=target_modules, seed=seed,
                    run_dir=out_root / "checkpoints" / f"{task}" / f"adalora_th{th:.2f}",
                    budget_total_rank=final_budget
                )
                adalora_final_accs[th] = ada_out["best_acc"]
            else:
                adalora_final_accs[th] = np.nan

        # -------- per-dataset plots --------
        plot_bayes_mean_dynamics(
            task=task,
            out_dir=plots_dir,
            thresholds=args.log_alpha_thresholds,
            per_threshold_bayes_rank_logs_by_seed=per_threshold_bayes_rank_logs_by_seed,
            per_threshold_bayes_acc_logs_by_seed=per_threshold_bayes_acc_logs_by_seed,
        )
        plot_final_acc_comparison(
            task=task,
            out_dir=plots_dir,
            thresholds=args.log_alpha_thresholds,
            bayes_results=per_threshold_final_bayes_accs,
            adalora_results=adalora_final_accs,
        )

        # -------- write results CSV --------
        rows = []
        for th in args.log_alpha_thresholds:
            rows.append({
                "task": task,
                "method": "BayesLoRA",
                "threshold": th,
                "final_acc_mean": float(np.mean(per_threshold_final_bayes_accs.get(th, []))) if per_threshold_final_bayes_accs.get(th) else np.nan,
                "final_acc_std":  float(np.std (per_threshold_final_bayes_accs.get(th, []))) if per_threshold_final_bayes_accs.get(th) else np.nan,
                "adalora_final_acc": float(adalora_final_accs.get(th, np.nan)),
                "final_budget_bayes_median": int(per_threshold_final_budget.get(th, 0)),
            })
        df = pd.DataFrame(rows)
        results_path = results_dir / f"{task}_summary.csv"
        df.to_csv(results_path, index=False)
        print(f"[{task}] summary written to {results_path}", flush=True)


if __name__ == "__main__":
    main()