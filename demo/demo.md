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