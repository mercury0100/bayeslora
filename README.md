# BayesLoRA

`python run_bayeslora.py \
  --freeze-base \
  --rank 64 --lora-alpha 128 \
  --local-reparam \
  --epochs 50 \
  --kl-freeze-epochs 25 \
  --beta-warmup-epochs 10 \
  --sample-warmup-epochs 15 \
  --kl-scale 0.001 \
  --logalpha-thresh 3.5 \
  --lr 5e-4`