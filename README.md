# BayesLoRA

## Quick variational dropout demo

```
python run_bayeslora.py \
  --freeze-base \
  --rank 168 --lora-alpha 256 \
  --local-reparam \
  --epochs 50 \
  --kl-freeze-epochs 25 \
  --beta-warmup-epochs 10 \
  --sample-warmup-epochs 15 \
  --kl-scale 0.05 \
  --logalpha-thresh 3 \
  --lr 5e-4
```
  
## Quick BayesLoRA demo:

```
python run_bayeslora_auto.py \
  --prune-every 5 \
  --logalpha-thresh 3.0 \
  --min-ranks-per-layer 2 \
  --freeze-base \
  --rank 168 --lora-alpha 256 \
  --local-reparam \
  --epochs 100 \
  --kl-freeze-epochs 25 \
  --beta-warmup-epochs 10 \
  --sample-warmup-epochs 15 \
  --kl-scale 0.05 \
  --logalpha-thresh 3 \
  --lr 5e-4
```

## BayesLoRA
```
⚡ master ~/bayeslora python benchmark_resnet_cifar.py --epochs 20 --rank 128 --prune-every 3 --min-ranks 2 --kl-scale 0.1

=== VLORA run (pretrained base) ===
Epoch 01 | loss 0.4186 | ce 0.4186 | kl 652.3962 | β 0.00 | sample 0 | acc_mean 91.31% | acc_mc 91.31%
Epoch 02 | loss 0.2026 | ce 0.2026 | kl 658.3311 | β 0.00 | sample 0 | acc_mean 92.23% | acc_mc 92.30%
Epoch 03 | loss 0.1577 | ce 0.1577 | kl 662.9644 | β 0.00 | sample 0 | acc_mean 93.88% | acc_mc 93.81%
Epoch 04 | loss 0.1807 | ce 0.1271 | kl 526.1633 | β 0.20 | sample 1 | acc_mean 93.45% | acc_mc 93.44%
Epoch 05 | loss 0.1726 | ce 0.1083 | kl 314.7952 | β 0.40 | sample 1 | acc_mean 93.90% | acc_mc 93.90%
Epoch 06 | loss 0.1371 | ce 0.0887 | kl 158.2159 | β 0.60 | sample 1 | acc_mean 93.90% | acc_mc 93.80%
Epoch 07 | loss 0.1084 | ce 0.0765 | kl 78.1221 | β 0.80 | sample 1 | acc_mean 94.36% | acc_mc 94.33%
Epoch 08 | loss 0.0940 | ce 0.0715 | kl 43.9577 | β 1.00 | sample 1 | acc_mean 94.47% | acc_mc 94.47%
Epoch 09 | loss 0.0827 | ce 0.0678 | kl 29.1244 | β 1.00 | sample 1 | acc_mean 94.59% | acc_mc 94.56%
[Prune] total ranks: 128 -> 62
Epoch 10 | loss 0.0837 | ce 0.0730 | kl 20.9791 | β 1.00 | sample 1 | acc_mean 93.90% | acc_mc 93.95%
Epoch 11 | loss 0.0753 | ce 0.0656 | kl 18.8501 | β 1.00 | sample 1 | acc_mean 93.99% | acc_mc 94.01%
Epoch 12 | loss 0.0695 | ce 0.0607 | kl 17.1662 | β 1.00 | sample 1 | acc_mean 94.33% | acc_mc 94.33%
[Prune] total ranks: 62 -> 34
Epoch 13 | loss 0.0688 | ce 0.0617 | kl 13.8968 | β 1.00 | sample 1 | acc_mean 94.72% | acc_mc 94.73%
Epoch 14 | loss 0.0640 | ce 0.0572 | kl 13.3265 | β 1.00 | sample 1 | acc_mean 94.84% | acc_mc 94.84%
Epoch 15 | loss 0.0553 | ce 0.0486 | kl 12.9798 | β 1.00 | sample 1 | acc_mean 94.72% | acc_mc 94.72%
[Prune] total ranks: 34 -> 29
Epoch 16 | loss 0.0578 | ce 0.0518 | kl 11.8182 | β 1.00 | sample 1 | acc_mean 94.73% | acc_mc 94.77%
Epoch 17 | loss 0.0539 | ce 0.0481 | kl 11.4212 | β 1.00 | sample 1 | acc_mean 94.53% | acc_mc 94.47%
Epoch 18 | loss 0.0477 | ce 0.0420 | kl 11.1825 | β 1.00 | sample 1 | acc_mean 94.08% | acc_mc 94.01%
[Prune] total ranks: 29 -> 27
Epoch 19 | loss 0.0521 | ce 0.0466 | kl 10.7493 | β 1.00 | sample 1 | acc_mean 94.29% | acc_mc 94.24%
Epoch 20 | loss 0.0519 | ce 0.0466 | kl 10.4983 | β 1.00 | sample 1 | acc_mean 94.60% | acc_mc 94.51%
[VLORA] total time: 11.8 min
[VLORA] Final acc (mean): 94.60% | (MC): 94.52%

=== LORA run (pretrained base) ===
Epoch 01 | loss 0.4044 | ce 0.4044 | β 0.00 | sample 0 | acc_mean 90.39% | acc_mc 90.39%
Epoch 02 | loss 0.2071 | ce 0.2071 | β 0.00 | sample 0 | acc_mean 93.02% | acc_mc 93.02%
Epoch 03 | loss 0.1585 | ce 0.1585 | β 0.00 | sample 0 | acc_mean 93.63% | acc_mc 93.63%
Epoch 04 | loss 0.1319 | ce 0.1319 | β 0.00 | sample 0 | acc_mean 94.03% | acc_mc 94.03%
Epoch 05 | loss 0.1113 | ce 0.1113 | β 0.00 | sample 0 | acc_mean 93.66% | acc_mc 93.66%
Epoch 06 | loss 0.0978 | ce 0.0978 | β 0.00 | sample 0 | acc_mean 94.17% | acc_mc 94.17%
Epoch 07 | loss 0.0850 | ce 0.0850 | β 0.00 | sample 0 | acc_mean 93.65% | acc_mc 93.65%
Epoch 08 | loss 0.0763 | ce 0.0763 | β 0.00 | sample 0 | acc_mean 94.41% | acc_mc 94.41%
Epoch 09 | loss 0.0754 | ce 0.0754 | β 0.00 | sample 0 | acc_mean 93.73% | acc_mc 93.73%
Epoch 10 | loss 0.0667 | ce 0.0667 | β 0.00 | sample 0 | acc_mean 93.77% | acc_mc 93.77%
Epoch 11 | loss 0.0647 | ce 0.0647 | β 0.00 | sample 0 | acc_mean 94.37% | acc_mc 94.37%
Epoch 12 | loss 0.0580 | ce 0.0580 | β 0.00 | sample 0 | acc_mean 94.90% | acc_mc 94.90%
Epoch 13 | loss 0.0534 | ce 0.0534 | β 0.00 | sample 0 | acc_mean 94.59% | acc_mc 94.59%
Epoch 14 | loss 0.0529 | ce 0.0529 | β 0.00 | sample 0 | acc_mean 94.31% | acc_mc 94.31%
Epoch 15 | loss 0.0469 | ce 0.0469 | β 0.00 | sample 0 | acc_mean 94.55% | acc_mc 94.55%
Epoch 16 | loss 0.0494 | ce 0.0494 | β 0.00 | sample 0 | acc_mean 94.35% | acc_mc 94.35%
Epoch 17 | loss 0.0475 | ce 0.0475 | β 0.00 | sample 0 | acc_mean 94.96% | acc_mc 94.96%
Epoch 18 | loss 0.0437 | ce 0.0437 | β 0.00 | sample 0 | acc_mean 94.53% | acc_mc 94.53%
Epoch 19 | loss 0.0441 | ce 0.0441 | β 0.00 | sample 0 | acc_mean 94.76% | acc_mc 94.76%
Epoch 20 | loss 0.0371 | ce 0.0371 | β 0.00 | sample 0 | acc_mean 94.79% | acc_mc 94.79%
[LORA] total time: 10.3 min
[LORA] Final acc (mean): 94.79% | (MC): 94.79%

=== Summary ===
BayesLoRA acc (mean): 94.60% | MC: 94.52%
LoRA     acc (mean): 94.79% | MC: 94.79%
```