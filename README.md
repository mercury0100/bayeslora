# BayesLoRA

## GPT-2

```
python bayeslora_nanogpt.py \
  --method bayeslora \
  --kl-freeze-steps 200 \
  --beta-warmup-steps 200 \
  --kl-scale 1e-5 \
  --sample-warmup-steps 0 \
  --rank 32 --lora-alpha 32 \
  --prune-every 200 --logalpha-thresh 4.0 --min-ranks 2 \
  --diag-every 100
```

### Output

```
⚡ master ~/bayeslora ppython bayeslora_gpt.py \
  --method bayeslora \
  --kl-freeze-steps 200 \
  --beta-warmup-steps 200 \
  --kl-scale 1e-6 \
  --sample-warmup-steps 0 \
  --rank 32 --lora-alpha 32 \
  --prune-every 200 --logalpha-thresh 2.0 --min-ranks 0 \
  --diag-every 100
`torch_dtype` is deprecated! Use `dtype` instead!
cuda available: True
device: NVIDIA L40S
model device: cuda:0
[BayesLoRA] Wrapped 12 modules with VariationalLoRAWrapper (rank=32).
[BayesLoRA] targets=qvo, last_k=6
[Trainable BAYESLORA] 1,769,472/126,209,280 params (1.402%)
Train blocks: 9486 | Valid blocks: 980 | Block size: 256
[BayesLoRA] Trainable params: 1,769,472
[Train] batches/epoch=1185 | grad_accum=8 | ~steps/epoch=149 | max_steps=2000 | diag_every=100
step 20/2000 | loss 0.5156 | ce 4.1250 | kl 0.000000 | β 0.00 | 7.84s since last
step 40/2000 | loss 0.5117 | ce 4.0938 | kl 0.000000 | β 0.00 | 7.16s since last
step 60/2000 | loss 0.5195 | ce 4.1562 | kl 0.000000 | β 0.00 | 7.13s since last
step 80/2000 | loss 0.4688 | ce 3.7500 | kl 0.000000 | β 0.00 | 7.14s since last
step 100/2000 | loss 0.4648 | ce 3.7188 | kl 0.000000 | β 0.00 | 7.18s since last
[Diag] step 100/2000 | ppl(det) 37.94 | ppl(MC,5) 37.61 | tok-acc(det) 36.9% | ECE(det) 0.014 | ECE(MC,5) 0.015
step 120/2000 | loss 0.4336 | ce 3.4688 | kl 0.000000 | β 0.00 | 18.43s since last
step 140/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 7.04s since last
step 160/2000 | loss 0.4531 | ce 3.6250 | kl 0.000000 | β 0.00 | 7.08s since last
step 180/2000 | loss 0.4395 | ce 3.5156 | kl 0.000000 | β 0.00 | 6.99s since last
step 200/2000 | loss 0.4395 | ce 3.5156 | kl 0.000000 | β 0.00 | 7.00s since last
[Diag] step 200/2000 | ppl(det) 30.03 | ppl(MC,5) 29.90 | tok-acc(det) 39.8% | ECE(det) 0.010 | ECE(MC,5) 0.009
step 220/2000 | loss 0.4300 | ce 3.4375 | kl 0.002779 | β 0.10 | 18.21s since last
step 240/2000 | loss 0.4129 | ce 3.2969 | kl 0.005940 | β 0.20 | 7.01s since last
step 260/2000 | loss 0.4211 | ce 3.3594 | kl 0.009255 | β 0.29 | 7.01s since last
step 280/2000 | loss 0.4352 | ce 3.4688 | kl 0.012639 | β 0.40 | 7.22s since last
step 300/2000 | loss 0.4415 | ce 3.5156 | kl 0.016059 | β 0.49 | 7.16s since last
[Diag] step 300/2000 | ppl(det) 27.84 | ppl(MC,5) 27.81 | tok-acc(det) 41.1% | ECE(det) 0.007 | ECE(MC,5) 0.005
step 320/2000 | loss 0.4263 | ce 3.3906 | kl 0.019476 | β 0.59 | 18.32s since last
step 340/2000 | loss 0.4189 | ce 3.3281 | kl 0.022859 | β 0.69 | 7.04s since last
step 360/2000 | loss 0.4330 | ce 3.4375 | kl 0.026139 | β 0.80 | 7.07s since last
step 380/2000 | loss 0.3943 | ce 3.1250 | kl 0.029332 | β 0.90 | 7.08s since last
step 400/2000 | loss 0.4376 | ce 3.4688 | kl 0.032379 | β 0.99 | 7.04s since last
[Diag] step 400/2000 | ppl(det) 27.23 | ppl(MC,5) 27.23 | tok-acc(det) 41.5% | ECE(det) 0.007 | ECE(MC,5) 0.007
[Prune@400] adapter params: 1769472 -> 1769472  ranks: {'transformer.h.6.attn.c_attn': 32, 'transformer.h.6.attn.c_proj': 32, 'transformer.h.7.attn.c_attn': 32, 'transformer.h.7.attn.c_proj': 32, 'transformer.h.8.attn.c_attn': 32, 'transformer.h.8.attn.c_proj': 32, 'transformer.h.9.attn.c_attn': 32, 'transformer.h.9.attn.c_proj': 32, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 32, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 32}
step 420/2000 | loss 0.4103 | ce 3.2500 | kl 0.032254 | β 1.00 | 18.41s since last
step 440/2000 | loss 0.4181 | ce 3.3125 | kl 0.031983 | β 1.00 | 6.97s since last
step 460/2000 | loss 0.4180 | ce 3.3125 | kl 0.031774 | β 1.00 | 7.13s since last
step 480/2000 | loss 0.4239 | ce 3.3594 | kl 0.031635 | β 1.00 | 6.99s since last
step 500/2000 | loss 0.4375 | ce 3.4688 | kl 0.031442 | β 1.00 | 7.03s since last
[Diag] step 500/2000 | ppl(det) 26.87 | ppl(MC,5) 26.86 | tok-acc(det) 41.8% | ECE(det) 0.006 | ECE(MC,5) 0.005
step 520/2000 | loss 0.4160 | ce 3.2969 | kl 0.031298 | β 1.00 | 18.42s since last
step 540/2000 | loss 0.4101 | ce 3.2500 | kl 0.031180 | β 1.00 | 7.00s since last
step 560/2000 | loss 0.4258 | ce 3.3750 | kl 0.031101 | β 1.00 | 7.03s since last
step 580/2000 | loss 0.4218 | ce 3.3438 | kl 0.030957 | β 1.00 | 6.99s since last
step 600/2000 | loss 0.4218 | ce 3.3438 | kl 0.030876 | β 1.00 | 7.15s since last
[Diag] step 600/2000 | ppl(det) 26.68 | ppl(MC,5) 26.65 | tok-acc(det) 41.9% | ECE(det) 0.008 | ECE(MC,5) 0.007
[Prune@600] adapter params: 1769472 -> 1714176  ranks: {'transformer.h.6.attn.c_attn': 32, 'transformer.h.6.attn.c_proj': 31, 'transformer.h.7.attn.c_attn': 32, 'transformer.h.7.attn.c_proj': 25, 'transformer.h.8.attn.c_attn': 32, 'transformer.h.8.attn.c_proj': 30, 'transformer.h.9.attn.c_attn': 32, 'transformer.h.9.attn.c_proj': 27, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 29, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 32}
step 620/2000 | loss 0.4042 | ce 3.2031 | kl 0.030260 | β 1.00 | 18.15s since last
step 640/2000 | loss 0.4198 | ce 3.3281 | kl 0.030207 | β 1.00 | 7.21s since last
step 660/2000 | loss 0.4296 | ce 3.4062 | kl 0.030166 | β 1.00 | 7.03s since last
step 680/2000 | loss 0.4335 | ce 3.4375 | kl 0.030145 | β 1.00 | 7.06s since last
step 700/2000 | loss 0.4139 | ce 3.2812 | kl 0.030106 | β 1.00 | 7.08s since last
[Diag] step 700/2000 | ppl(det) 26.52 | ppl(MC,5) 26.50 | tok-acc(det) 42.0% | ECE(det) 0.007 | ECE(MC,5) 0.005
step 720/2000 | loss 0.4100 | ce 3.2500 | kl 0.030094 | β 1.00 | 18.19s since last
step 740/2000 | loss 0.4120 | ce 3.2656 | kl 0.030032 | β 1.00 | 7.02s since last
step 760/2000 | loss 0.4061 | ce 3.2188 | kl 0.030022 | β 1.00 | 7.13s since last
step 780/2000 | loss 0.4373 | ce 3.4688 | kl 0.030032 | β 1.00 | 7.06s since last
step 800/2000 | loss 0.3748 | ce 2.9688 | kl 0.030039 | β 1.00 | 7.21s since last
[Diag] step 800/2000 | ppl(det) 26.39 | ppl(MC,5) 26.35 | tok-acc(det) 42.1% | ECE(det) 0.009 | ECE(MC,5) 0.007
[Prune@800] adapter params: 1714176 -> 1520640  ranks: {'transformer.h.6.attn.c_attn': 32, 'transformer.h.6.attn.c_proj': 26, 'transformer.h.7.attn.c_attn': 29, 'transformer.h.7.attn.c_proj': 21, 'transformer.h.8.attn.c_attn': 27, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 32, 'transformer.h.9.attn.c_proj': 20, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 21, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 25}
step 820/2000 | loss 0.4371 | ce 3.4688 | kl 0.027942 | β 1.00 | 18.29s since last
step 840/2000 | loss 0.4391 | ce 3.4844 | kl 0.028054 | β 1.00 | 7.00s since last
step 860/2000 | loss 0.4098 | ce 3.2500 | kl 0.028121 | β 1.00 | 7.03s since last
step 880/2000 | loss 0.3883 | ce 3.0781 | kl 0.028190 | β 1.00 | 6.99s since last
step 900/2000 | loss 0.4352 | ce 3.4531 | kl 0.028222 | β 1.00 | 7.15s since last
[Diag] step 900/2000 | ppl(det) 26.33 | ppl(MC,5) 26.29 | tok-acc(det) 42.1% | ECE(det) 0.007 | ECE(MC,5) 0.006
step 920/2000 | loss 0.3590 | ce 2.8438 | kl 0.028273 | β 1.00 | 18.44s since last
step 940/2000 | loss 0.4371 | ce 3.4688 | kl 0.028326 | β 1.00 | 7.06s since last
step 960/2000 | loss 0.4078 | ce 3.2344 | kl 0.028356 | β 1.00 | 7.07s since last
step 980/2000 | loss 0.4254 | ce 3.3750 | kl 0.028381 | β 1.00 | 7.06s since last
step 1000/2000 | loss 0.4039 | ce 3.2031 | kl 0.028431 | β 1.00 | 7.11s since last
[Diag] step 1000/2000 | ppl(det) 26.23 | ppl(MC,5) 26.21 | tok-acc(det) 42.1% | ECE(det) 0.008 | ECE(MC,5) 0.006
[Prune@1000] adapter params: 1520640 -> 1492992  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 23, 'transformer.h.7.attn.c_attn': 28, 'transformer.h.7.attn.c_proj': 20, 'transformer.h.8.attn.c_attn': 27, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 32, 'transformer.h.9.attn.c_proj': 19, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 21, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 25}
step 1020/2000 | loss 0.3844 | ce 3.0469 | kl 0.028144 | β 1.00 | 18.55s since last
step 1040/2000 | loss 0.4117 | ce 3.2656 | kl 0.028187 | β 1.00 | 7.17s since last
step 1060/2000 | loss 0.4078 | ce 3.2344 | kl 0.028226 | β 1.00 | 7.09s since last
step 1080/2000 | loss 0.4078 | ce 3.2344 | kl 0.028259 | β 1.00 | 7.10s since last
step 1100/2000 | loss 0.3922 | ce 3.1094 | kl 0.028299 | β 1.00 | 7.04s since last
[Diag] step 1100/2000 | ppl(det) 26.23 | ppl(MC,5) 26.21 | tok-acc(det) 42.2% | ECE(det) 0.009 | ECE(MC,5) 0.007
step 1120/2000 | loss 0.4117 | ce 3.2656 | kl 0.028333 | β 1.00 | 18.35s since last
step 1140/2000 | loss 0.3903 | ce 3.0938 | kl 0.028360 | β 1.00 | 7.09s since last
step 1160/2000 | loss 0.4196 | ce 3.3281 | kl 0.028372 | β 1.00 | 7.23s since last
step 1180/2000 | loss 0.4411 | ce 3.5000 | kl 0.028410 | β 1.00 | 7.05s since last
step 1200/2000 | loss 0.4000 | ce 3.1719 | kl 0.028429 | β 1.00 | 7.20s since last
[Diag] step 1200/2000 | ppl(det) 26.14 | ppl(MC,5) 26.11 | tok-acc(det) 42.2% | ECE(det) 0.009 | ECE(MC,5) 0.008
[Prune@1200] adapter params: 1492992 -> 1462272  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 22, 'transformer.h.7.attn.c_attn': 28, 'transformer.h.7.attn.c_proj': 18, 'transformer.h.8.attn.c_attn': 25, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 32, 'transformer.h.9.attn.c_proj': 18, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 19, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 25}
step 1220/2000 | loss 0.4410 | ce 3.5000 | kl 0.028070 | β 1.00 | 18.38s since last
step 1240/2000 | loss 0.4137 | ce 3.2812 | kl 0.028115 | β 1.00 | 7.07s since last
step 1260/2000 | loss 0.4352 | ce 3.4531 | kl 0.028151 | β 1.00 | 7.04s since last
step 1280/2000 | loss 0.4234 | ce 3.3594 | kl 0.028172 | β 1.00 | 7.01s since last
step 1300/2000 | loss 0.3629 | ce 2.8750 | kl 0.028186 | β 1.00 | 7.26s since last
[Diag] step 1300/2000 | ppl(det) 26.12 | ppl(MC,5) 26.10 | tok-acc(det) 42.2% | ECE(det) 0.008 | ECE(MC,5) 0.006
step 1320/2000 | loss 0.4352 | ce 3.4531 | kl 0.028215 | β 1.00 | 18.19s since last
step 1340/2000 | loss 0.4254 | ce 3.3750 | kl 0.028232 | β 1.00 | 7.18s since last
step 1360/2000 | loss 0.4235 | ce 3.3594 | kl 0.028266 | β 1.00 | 7.08s since last
step 1380/2000 | loss 0.4039 | ce 3.2031 | kl 0.028282 | β 1.00 | 7.07s since last
step 1400/2000 | loss 0.4274 | ce 3.3906 | kl 0.028307 | β 1.00 | 7.06s since last
[Diag] step 1400/2000 | ppl(det) 26.10 | ppl(MC,5) 26.06 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.008
[Prune@1400] adapter params: 1462272 -> 1453056  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 22, 'transformer.h.7.attn.c_attn': 28, 'transformer.h.7.attn.c_proj': 18, 'transformer.h.8.attn.c_attn': 25, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 31, 'transformer.h.9.attn.c_proj': 18, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 19, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 24}
step 1420/2000 | loss 0.3766 | ce 2.9844 | kl 0.028221 | β 1.00 | 18.58s since last
step 1440/2000 | loss 0.4020 | ce 3.1875 | kl 0.028242 | β 1.00 | 7.14s since last
step 1460/2000 | loss 0.4528 | ce 3.5938 | kl 0.028259 | β 1.00 | 7.04s since last
step 1480/2000 | loss 0.3922 | ce 3.1094 | kl 0.028282 | β 1.00 | 7.01s since last
step 1500/2000 | loss 0.4196 | ce 3.3281 | kl 0.028288 | β 1.00 | 7.17s since last
[Diag] step 1500/2000 | ppl(det) 26.04 | ppl(MC,5) 26.01 | tok-acc(det) 42.2% | ECE(det) 0.010 | ECE(MC,5) 0.008
step 1520/2000 | loss 0.4078 | ce 3.2344 | kl 0.028310 | β 1.00 | 18.59s since last
step 1540/2000 | loss 0.4098 | ce 3.2500 | kl 0.028335 | β 1.00 | 7.01s since last
step 1560/2000 | loss 0.4137 | ce 3.2812 | kl 0.028349 | β 1.00 | 7.08s since last
step 1580/2000 | loss 0.4117 | ce 3.2656 | kl 0.028367 | β 1.00 | 7.00s since last
step 1600/2000 | loss 0.4039 | ce 3.2031 | kl 0.028366 | β 1.00 | 7.00s since last
[Diag] step 1600/2000 | ppl(det) 26.06 | ppl(MC,5) 26.00 | tok-acc(det) 42.2% | ECE(det) 0.008 | ECE(MC,5) 0.007
[Prune@1600] adapter params: 1453056 -> 1431552  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 21, 'transformer.h.7.attn.c_attn': 27, 'transformer.h.7.attn.c_proj': 18, 'transformer.h.8.attn.c_attn': 25, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 30, 'transformer.h.9.attn.c_proj': 17, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 18, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 24}
step 1620/2000 | loss 0.4312 | ce 3.4219 | kl 0.028113 | β 1.00 | 18.20s since last
step 1640/2000 | loss 0.4195 | ce 3.3281 | kl 0.028127 | β 1.00 | 7.13s since last
step 1660/2000 | loss 0.4371 | ce 3.4688 | kl 0.028141 | β 1.00 | 7.09s since last
step 1680/2000 | loss 0.4078 | ce 3.2344 | kl 0.028154 | β 1.00 | 7.31s since last
step 1700/2000 | loss 0.4098 | ce 3.2500 | kl 0.028164 | β 1.00 | 7.07s since last
[Diag] step 1700/2000 | ppl(det) 26.03 | ppl(MC,5) 25.98 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.007
step 1720/2000 | loss 0.3922 | ce 3.1094 | kl 0.028181 | β 1.00 | 18.28s since last
step 1740/2000 | loss 0.4215 | ce 3.3438 | kl 0.028194 | β 1.00 | 7.05s since last
step 1760/2000 | loss 0.4195 | ce 3.3281 | kl 0.028208 | β 1.00 | 7.07s since last
step 1780/2000 | loss 0.4215 | ce 3.3438 | kl 0.028214 | β 1.00 | 7.14s since last
step 1800/2000 | loss 0.4039 | ce 3.2031 | kl 0.028218 | β 1.00 | 7.08s since last
[Diag] step 1800/2000 | ppl(det) 26.01 | ppl(MC,5) 25.97 | tok-acc(det) 42.2% | ECE(det) 0.009 | ECE(MC,5) 0.006
[Prune@1800] adapter params: 1431552 -> 1431552  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 21, 'transformer.h.7.attn.c_attn': 27, 'transformer.h.7.attn.c_proj': 18, 'transformer.h.8.attn.c_attn': 25, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 30, 'transformer.h.9.attn.c_proj': 17, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 18, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 24}
step 1820/2000 | loss 0.4020 | ce 3.1875 | kl 0.028231 | β 1.00 | 18.51s since last
step 1840/2000 | loss 0.4430 | ce 3.5156 | kl 0.028237 | β 1.00 | 7.13s since last
step 1860/2000 | loss 0.4215 | ce 3.3438 | kl 0.028239 | β 1.00 | 7.08s since last
step 1880/2000 | loss 0.3883 | ce 3.0781 | kl 0.028242 | β 1.00 | 7.09s since last
step 1900/2000 | loss 0.4137 | ce 3.2812 | kl 0.028245 | β 1.00 | 7.09s since last
[Diag] step 1900/2000 | ppl(det) 26.01 | ppl(MC,5) 25.96 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.006
step 1920/2000 | loss 0.4195 | ce 3.3281 | kl 0.028250 | β 1.00 | 18.45s since last
step 1940/2000 | loss 0.4137 | ce 3.2812 | kl 0.028252 | β 1.00 | 7.18s since last
step 1960/2000 | loss 0.3981 | ce 3.1562 | kl 0.028253 | β 1.00 | 7.06s since last
step 1980/2000 | loss 0.4391 | ce 3.4844 | kl 0.028255 | β 1.00 | 7.03s since last
step 2000/2000 | loss 0.4254 | ce 3.3750 | kl 0.028255 | β 1.00 | 7.02s since last
[Diag] step 2000/2000 | ppl(det) 26.00 | ppl(MC,5) 25.96 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.007
[Prune@2000] adapter params: 1431552 -> 1431552  ranks: {'transformer.h.6.attn.c_attn': 31, 'transformer.h.6.attn.c_proj': 21, 'transformer.h.7.attn.c_attn': 27, 'transformer.h.7.attn.c_proj': 18, 'transformer.h.8.attn.c_attn': 25, 'transformer.h.8.attn.c_proj': 14, 'transformer.h.9.attn.c_attn': 30, 'transformer.h.9.attn.c_proj': 17, 'transformer.h.10.attn.c_attn': 32, 'transformer.h.10.attn.c_proj': 18, 'transformer.h.11.attn.c_attn': 32, 'transformer.h.11.attn.c_proj': 24}

=== Results (WikiText-2) ===
Method: bayeslora
Perplexity (det): 26.001 | Perplexity (MC, 5): 25.947
Token-Acc (det): 40.27% | ECE(det): 0.0133
Token-Acc (MC):  40.29% | ECE(MC,5): 0.0114


⚡ master ~/bayeslora ppython bayeslora_gpt.py \
  --method lora \     
  --kl-freeze-steps 200 \
  --beta-warmup-steps 200 \
  --kl-scale 1e-5 \
  --sample-warmup-steps 0 \
  --rank 32 --lora-alpha 32 \
  --prune-every 200 --logalpha-thresh 4.0 --min-ranks 2 \
  --diag-every 100
`torch_dtype` is deprecated! Use `dtype` instead!
cuda available: True
device: NVIDIA L40S
model device: cuda:0
[LoRA] Wrapped 12 GPT-2 modules (rank=32, targets=qvo, last_k=6).
[Trainable LORA] 884,736/125,324,544 params (0.706%)
Train blocks: 9486 | Valid blocks: 980 | Block size: 256
[Train] batches/epoch=1185 | grad_accum=8 | ~steps/epoch=149 | max_steps=2000 | diag_every=100
step 20/2000 | loss 0.5156 | ce 4.1250 | kl 0.000000 | β 0.00 | 4.00s since last
step 40/2000 | loss 0.5117 | ce 4.0938 | kl 0.000000 | β 0.00 | 3.27s since last
step 60/2000 | loss 0.5234 | ce 4.1875 | kl 0.000000 | β 0.00 | 3.27s since last
step 80/2000 | loss 0.4727 | ce 3.7812 | kl 0.000000 | β 0.00 | 3.29s since last
step 100/2000 | loss 0.4629 | ce 3.7031 | kl 0.000000 | β 0.00 | 3.28s since last
[Diag] step 100/2000 | ppl(det) 37.83 | ppl(MC,5) 37.83 | tok-acc(det) 36.8% | ECE(det) 0.014 | ECE(MC,5) 0.014
step 120/2000 | loss 0.4355 | ce 3.4844 | kl 0.000000 | β 0.00 | 5.07s since last
step 140/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 3.22s since last
step 160/2000 | loss 0.4414 | ce 3.5312 | kl 0.000000 | β 0.00 | 3.31s since last
step 180/2000 | loss 0.4160 | ce 3.3281 | kl 0.000000 | β 0.00 | 3.28s since last
step 200/2000 | loss 0.4277 | ce 3.4219 | kl 0.000000 | β 0.00 | 3.45s since last
[Diag] step 200/2000 | ppl(det) 29.97 | ppl(MC,5) 29.97 | tok-acc(det) 39.9% | ECE(det) 0.010 | ECE(MC,5) 0.010
step 220/2000 | loss 0.4316 | ce 3.4531 | kl 0.000000 | β 0.00 | 5.04s since last
step 240/2000 | loss 0.4414 | ce 3.5312 | kl 0.000000 | β 0.00 | 3.24s since last
step 260/2000 | loss 0.4180 | ce 3.3438 | kl 0.000000 | β 0.00 | 3.27s since last
step 280/2000 | loss 0.4336 | ce 3.4688 | kl 0.000000 | β 0.00 | 3.23s since last
step 300/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.34s since last
[Diag] step 300/2000 | ppl(det) 27.79 | ppl(MC,5) 27.79 | tok-acc(det) 41.2% | ECE(det) 0.007 | ECE(MC,5) 0.007
step 320/2000 | loss 0.4277 | ce 3.4219 | kl 0.000000 | β 0.00 | 5.17s since last
step 340/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.25s since last
step 360/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.26s since last
step 380/2000 | loss 0.4082 | ce 3.2656 | kl 0.000000 | β 0.00 | 3.26s since last
step 400/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.46s since last
[Diag] step 400/2000 | ppl(det) 27.17 | ppl(MC,5) 27.17 | tok-acc(det) 41.6% | ECE(det) 0.008 | ECE(MC,5) 0.008
step 420/2000 | loss 0.4355 | ce 3.4844 | kl 0.000000 | β 0.00 | 5.06s since last
step 440/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.22s since last
step 460/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.32s since last
step 480/2000 | loss 0.4043 | ce 3.2344 | kl 0.000000 | β 0.00 | 3.22s since last
step 500/2000 | loss 0.4160 | ce 3.3281 | kl 0.000000 | β 0.00 | 3.25s since last
[Diag] step 500/2000 | ppl(det) 26.78 | ppl(MC,5) 26.78 | tok-acc(det) 41.8% | ECE(det) 0.006 | ECE(MC,5) 0.006
step 520/2000 | loss 0.4336 | ce 3.4688 | kl 0.000000 | β 0.00 | 5.10s since last
step 540/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.26s since last
step 560/2000 | loss 0.3867 | ce 3.0938 | kl 0.000000 | β 0.00 | 3.26s since last
step 580/2000 | loss 0.3945 | ce 3.1562 | kl 0.000000 | β 0.00 | 3.23s since last
step 600/2000 | loss 0.3887 | ce 3.1094 | kl 0.000000 | β 0.00 | 3.36s since last
[Diag] step 600/2000 | ppl(det) 26.58 | ppl(MC,5) 26.58 | tok-acc(det) 42.0% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 620/2000 | loss 0.3926 | ce 3.1406 | kl 0.000000 | β 0.00 | 5.26s since last
step 640/2000 | loss 0.4297 | ce 3.4375 | kl 0.000000 | β 0.00 | 3.17s since last
step 660/2000 | loss 0.4316 | ce 3.4531 | kl 0.000000 | β 0.00 | 3.16s since last
step 680/2000 | loss 0.3945 | ce 3.1562 | kl 0.000000 | β 0.00 | 3.14s since last
step 700/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.18s since last
[Diag] step 700/2000 | ppl(det) 26.41 | ppl(MC,5) 26.41 | tok-acc(det) 42.1% | ECE(det) 0.010 | ECE(MC,5) 0.010
step 720/2000 | loss 0.4160 | ce 3.3281 | kl 0.000000 | β 0.00 | 5.10s since last
step 740/2000 | loss 0.4238 | ce 3.3906 | kl 0.000000 | β 0.00 | 3.25s since last
step 760/2000 | loss 0.4316 | ce 3.4531 | kl 0.000000 | β 0.00 | 3.28s since last
step 780/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.21s since last
step 800/2000 | loss 0.4238 | ce 3.3906 | kl 0.000000 | β 0.00 | 3.23s since last
[Diag] step 800/2000 | ppl(det) 26.30 | ppl(MC,5) 26.30 | tok-acc(det) 42.2% | ECE(det) 0.007 | ECE(MC,5) 0.007
step 820/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 5.27s since last
step 840/2000 | loss 0.4043 | ce 3.2344 | kl 0.000000 | β 0.00 | 3.23s since last
step 860/2000 | loss 0.4023 | ce 3.2188 | kl 0.000000 | β 0.00 | 3.20s since last
step 880/2000 | loss 0.4043 | ce 3.2344 | kl 0.000000 | β 0.00 | 3.22s since last
step 900/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.36s since last
[Diag] step 900/2000 | ppl(det) 26.20 | ppl(MC,5) 26.20 | tok-acc(det) 42.2% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 920/2000 | loss 0.4082 | ce 3.2656 | kl 0.000000 | β 0.00 | 5.15s since last
step 940/2000 | loss 0.4219 | ce 3.3750 | kl 0.000000 | β 0.00 | 3.27s since last
step 960/2000 | loss 0.4082 | ce 3.2656 | kl 0.000000 | β 0.00 | 3.26s since last
step 980/2000 | loss 0.3770 | ce 3.0156 | kl 0.000000 | β 0.00 | 3.24s since last
step 1000/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 3.22s since last
[Diag] step 1000/2000 | ppl(det) 26.13 | ppl(MC,5) 26.13 | tok-acc(det) 42.3% | ECE(det) 0.007 | ECE(MC,5) 0.007
step 1020/2000 | loss 0.3887 | ce 3.1094 | kl 0.000000 | β 0.00 | 5.16s since last
step 1040/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.27s since last
step 1060/2000 | loss 0.4180 | ce 3.3438 | kl 0.000000 | β 0.00 | 3.20s since last
step 1080/2000 | loss 0.4395 | ce 3.5156 | kl 0.000000 | β 0.00 | 3.20s since last
step 1100/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.20s since last
[Diag] step 1100/2000 | ppl(det) 26.05 | ppl(MC,5) 26.05 | tok-acc(det) 42.4% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1120/2000 | loss 0.3828 | ce 3.0625 | kl 0.000000 | β 0.00 | 5.03s since last
step 1140/2000 | loss 0.4336 | ce 3.4688 | kl 0.000000 | β 0.00 | 3.23s since last
step 1160/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 3.23s since last
step 1180/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 3.26s since last
step 1200/2000 | loss 0.4160 | ce 3.3281 | kl 0.000000 | β 0.00 | 3.32s since last
[Diag] step 1200/2000 | ppl(det) 25.99 | ppl(MC,5) 25.99 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1220/2000 | loss 0.4141 | ce 3.3125 | kl 0.000000 | β 0.00 | 5.28s since last
step 1240/2000 | loss 0.3965 | ce 3.1719 | kl 0.000000 | β 0.00 | 3.28s since last
step 1260/2000 | loss 0.4316 | ce 3.4531 | kl 0.000000 | β 0.00 | 3.24s since last
step 1280/2000 | loss 0.4062 | ce 3.2500 | kl 0.000000 | β 0.00 | 3.24s since last
step 1300/2000 | loss 0.4023 | ce 3.2188 | kl 0.000000 | β 0.00 | 3.23s since last
[Diag] step 1300/2000 | ppl(det) 25.95 | ppl(MC,5) 25.95 | tok-acc(det) 42.4% | ECE(det) 0.008 | ECE(MC,5) 0.008
step 1320/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 5.10s since last
step 1340/2000 | loss 0.3887 | ce 3.1094 | kl 0.000000 | β 0.00 | 3.27s since last
step 1360/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.24s since last
step 1380/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.23s since last
step 1400/2000 | loss 0.3750 | ce 3.0000 | kl 0.000000 | β 0.00 | 3.26s since last
[Diag] step 1400/2000 | ppl(det) 25.90 | ppl(MC,5) 25.90 | tok-acc(det) 42.5% | ECE(det) 0.007 | ECE(MC,5) 0.007
step 1420/2000 | loss 0.3984 | ce 3.1875 | kl 0.000000 | β 0.00 | 5.09s since last
step 1440/2000 | loss 0.3984 | ce 3.1875 | kl 0.000000 | β 0.00 | 3.41s since last
step 1460/2000 | loss 0.4297 | ce 3.4375 | kl 0.000000 | β 0.00 | 3.21s since last
step 1480/2000 | loss 0.4141 | ce 3.3125 | kl 0.000000 | β 0.00 | 3.21s since last
step 1500/2000 | loss 0.4219 | ce 3.3750 | kl 0.000000 | β 0.00 | 3.32s since last
[Diag] step 1500/2000 | ppl(det) 25.89 | ppl(MC,5) 25.89 | tok-acc(det) 42.4% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1520/2000 | loss 0.4219 | ce 3.3750 | kl 0.000000 | β 0.00 | 5.07s since last
step 1540/2000 | loss 0.4004 | ce 3.2031 | kl 0.000000 | β 0.00 | 3.22s since last
step 1560/2000 | loss 0.4043 | ce 3.2344 | kl 0.000000 | β 0.00 | 3.22s since last
step 1580/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.20s since last
step 1600/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.22s since last
[Diag] step 1600/2000 | ppl(det) 25.87 | ppl(MC,5) 25.87 | tok-acc(det) 42.5% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1620/2000 | loss 0.4023 | ce 3.2188 | kl 0.000000 | β 0.00 | 5.07s since last
step 1640/2000 | loss 0.3809 | ce 3.0469 | kl 0.000000 | β 0.00 | 3.50s since last
step 1660/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.21s since last
step 1680/2000 | loss 0.3965 | ce 3.1719 | kl 0.000000 | β 0.00 | 3.22s since last
step 1700/2000 | loss 0.4258 | ce 3.4062 | kl 0.000000 | β 0.00 | 3.23s since last
[Diag] step 1700/2000 | ppl(det) 25.84 | ppl(MC,5) 25.84 | tok-acc(det) 42.3% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1720/2000 | loss 0.3867 | ce 3.0938 | kl 0.000000 | β 0.00 | 5.12s since last
step 1740/2000 | loss 0.4473 | ce 3.5781 | kl 0.000000 | β 0.00 | 3.25s since last
step 1760/2000 | loss 0.4336 | ce 3.4688 | kl 0.000000 | β 0.00 | 3.25s since last
step 1780/2000 | loss 0.3574 | ce 2.8594 | kl 0.000000 | β 0.00 | 3.34s since last
step 1800/2000 | loss 0.4121 | ce 3.2969 | kl 0.000000 | β 0.00 | 3.23s since last
[Diag] step 1800/2000 | ppl(det) 25.81 | ppl(MC,5) 25.81 | tok-acc(det) 42.5% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1820/2000 | loss 0.3848 | ce 3.0781 | kl 0.000000 | β 0.00 | 5.08s since last
step 1840/2000 | loss 0.3906 | ce 3.1250 | kl 0.000000 | β 0.00 | 3.42s since last
step 1860/2000 | loss 0.4102 | ce 3.2812 | kl 0.000000 | β 0.00 | 3.20s since last
step 1880/2000 | loss 0.4062 | ce 3.2500 | kl 0.000000 | β 0.00 | 3.21s since last
step 1900/2000 | loss 0.3652 | ce 2.9219 | kl 0.000000 | β 0.00 | 3.20s since last
[Diag] step 1900/2000 | ppl(det) 25.86 | ppl(MC,5) 25.86 | tok-acc(det) 42.4% | ECE(det) 0.009 | ECE(MC,5) 0.009
step 1920/2000 | loss 0.4219 | ce 3.3750 | kl 0.000000 | β 0.00 | 5.19s since last
step 1940/2000 | loss 0.3887 | ce 3.1094 | kl 0.000000 | β 0.00 | 3.34s since last
step 1960/2000 | loss 0.4043 | ce 3.2344 | kl 0.000000 | β 0.00 | 3.24s since last
step 1980/2000 | loss 0.4297 | ce 3.4375 | kl 0.000000 | β 0.00 | 3.25s since last
step 2000/2000 | loss 0.4199 | ce 3.3594 | kl 0.000000 | β 0.00 | 3.26s since last
[Diag] step 2000/2000 | ppl(det) 25.84 | ppl(MC,5) 25.84 | tok-acc(det) 42.5% | ECE(det) 0.009 | ECE(MC,5) 0.009

=== Results (WikiText-2) ===
Method: lora
Perplexity (det): 25.836 | Perplexity (MC, 5): 25.836
Token-Acc (det): 40.34% | ECE(det): 0.0140
Token-Acc (MC):  40.34% | ECE(MC,5): 0.0140
```