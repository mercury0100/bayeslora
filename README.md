# BayesLoRA

## GPT-2

```
⚡ master ~/bayeslora python bayeslora_gpt.py \
  --method bayeslora \
  --include-mlp \
  --kl-scale 1e-6 \
  --rank 32 --lora-alpha 32 \
  --prune-every 200 --logalpha-thresh 2.0 \
  --diag-every 100 --last-k-layers 12
cuda available: True
device: NVIDIA H100 80GB HBM3
model device: cuda:0
[BayesLoRA] Wrapped 48 modules with VariationalLoRAWrapper (rank=32).
[BayesLoRA] targets=qvo, last_k=12, include_mlp=True
[Trainable BAYESLORA] 4,720,128/129,159,936 params (3.654%)
Train blocks: 9486 | Valid blocks: 980 | Block size: 256
[BayesLoRA] Trainable params: 4,720,128
[Train] batches/epoch=1185 | grad_accum=8 | ~steps/epoch=149 | max_steps=2000 | diag_every=100
step 20/2000 | loss 0.5248 | ce 4.1250 | kl 0.073187 | 18.77s since last
step 40/2000 | loss 0.5070 | ce 3.9844 | kl 0.071800 | 18.34s since last
step 60/2000 | loss 0.5050 | ce 3.9688 | kl 0.071501 | 18.21s since last
step 80/2000 | loss 0.4581 | ce 3.5938 | kl 0.070985 | 18.26s since last
step 100/2000 | loss 0.4463 | ce 3.5000 | kl 0.070348 | 18.17s since last
[Diag] step 100/2000 | ppl(det) 32.75 | ppl(MC,5) 32.49 | tok-acc(det) 38.7% | ECE(det) 0.013 | ECE(MC,5) 0.013
step 120/2000 | loss 0.4227 | ce 3.3125 | kl 0.069261 | 34.39s since last
step 140/2000 | loss 0.4107 | ce 3.2188 | kl 0.067184 | 17.99s since last
step 160/2000 | loss 0.4377 | ce 3.4375 | kl 0.064065 | 18.28s since last
step 180/2000 | loss 0.4294 | ce 3.3750 | kl 0.060574 | 18.05s since last
step 200/2000 | loss 0.4309 | ce 3.3906 | kl 0.056834 | 18.01s since last
[Diag] step 200/2000 | ppl(det) 26.13 | ppl(MC,5) 26.01 | tok-acc(det) 42.1% | ECE(det) 0.010 | ECE(MC,5) 0.005
[Prune@200] adapter params: 4720128 -> 924918  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 29, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 31, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 32, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 32, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 220/2000 | loss 0.4300 | ce 3.4219 | kl 0.017771 | 25.09s since last
step 240/2000 | loss 0.4125 | ce 3.2812 | kl 0.019119 | 9.07s since last
step 260/2000 | loss 0.4146 | ce 3.2969 | kl 0.020033 | 8.97s since last
step 280/2000 | loss 0.4284 | ce 3.4062 | kl 0.020731 | 8.94s since last
step 300/2000 | loss 0.4363 | ce 3.4688 | kl 0.021303 | 9.04s since last
[Diag] step 300/2000 | ppl(det) 26.47 | ppl(MC,5) 26.34 | tok-acc(det) 41.8% | ECE(det) 0.011 | ECE(MC,5) 0.008
step 320/2000 | loss 0.4207 | ce 3.3438 | kl 0.021809 | 20.02s since last
step 340/2000 | loss 0.4110 | ce 3.2656 | kl 0.022277 | 8.88s since last
step 360/2000 | loss 0.4247 | ce 3.3750 | kl 0.022637 | 8.90s since last
step 380/2000 | loss 0.3818 | ce 3.0312 | kl 0.022948 | 8.92s since last
step 400/2000 | loss 0.4287 | ce 3.4062 | kl 0.023219 | 9.05s since last
[Diag] step 400/2000 | ppl(det) 25.76 | ppl(MC,5) 25.66 | tok-acc(det) 42.3% | ECE(det) 0.011 | ECE(MC,5) 0.007
[Prune@400] adapter params: 924918 -> 924918  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 29, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 31, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 32, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 32, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 420/2000 | loss 0.4033 | ce 3.2031 | kl 0.023444 | 19.90s since last
step 440/2000 | loss 0.4092 | ce 3.2500 | kl 0.023651 | 8.91s since last
step 460/2000 | loss 0.4112 | ce 3.2656 | kl 0.023868 | 9.00s since last
step 480/2000 | loss 0.4151 | ce 3.2969 | kl 0.024116 | 8.89s since last
step 500/2000 | loss 0.4269 | ce 3.3906 | kl 0.024249 | 8.95s since last
[Diag] step 500/2000 | ppl(det) 25.40 | ppl(MC,5) 25.32 | tok-acc(det) 42.5% | ECE(det) 0.011 | ECE(MC,5) 0.008
step 520/2000 | loss 0.4054 | ce 3.2188 | kl 0.024404 | 20.23s since last
step 540/2000 | loss 0.4015 | ce 3.1875 | kl 0.024546 | 8.94s since last
step 560/2000 | loss 0.4191 | ce 3.3281 | kl 0.024675 | 8.88s since last
step 580/2000 | loss 0.4113 | ce 3.2656 | kl 0.024787 | 8.90s since last
step 600/2000 | loss 0.4113 | ce 3.2656 | kl 0.024896 | 9.03s since last
[Diag] step 600/2000 | ppl(det) 25.16 | ppl(MC,5) 25.12 | tok-acc(det) 42.6% | ECE(det) 0.009 | ECE(MC,5) 0.007
[Prune@600] adapter params: 924918 -> 905713  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 29, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 28, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 30, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 32, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 620/2000 | loss 0.3976 | ce 3.1562 | kl 0.024778 | 20.12s since last
step 640/2000 | loss 0.4094 | ce 3.2500 | kl 0.024885 | 8.87s since last
step 660/2000 | loss 0.4230 | ce 3.3594 | kl 0.024996 | 8.91s since last
step 680/2000 | loss 0.4270 | ce 3.3906 | kl 0.025078 | 8.86s since last
step 700/2000 | loss 0.4055 | ce 3.2188 | kl 0.025156 | 8.96s since last
[Diag] step 700/2000 | ppl(det) 24.99 | ppl(MC,5) 24.95 | tok-acc(det) 42.6% | ECE(det) 0.010 | ECE(MC,5) 0.007
step 720/2000 | loss 0.4016 | ce 3.1875 | kl 0.025231 | 20.20s since last
step 740/2000 | loss 0.4036 | ce 3.2031 | kl 0.025283 | 8.99s since last
step 760/2000 | loss 0.3957 | ce 3.1406 | kl 0.025356 | 9.22s since last
step 780/2000 | loss 0.4270 | ce 3.3906 | kl 0.025431 | 8.92s since last
step 800/2000 | loss 0.3684 | ce 2.9219 | kl 0.025516 | 8.92s since last
[Diag] step 800/2000 | ppl(det) 24.88 | ppl(MC,5) 24.83 | tok-acc(det) 42.8% | ECE(det) 0.012 | ECE(MC,5) 0.008
[Prune@800] adapter params: 905713 -> 840416  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 24, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 23, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 24, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 31, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 820/2000 | loss 0.4269 | ce 3.3906 | kl 0.024567 | 19.97s since last
step 840/2000 | loss 0.4328 | ce 3.4375 | kl 0.024696 | 8.95s since last
step 860/2000 | loss 0.4035 | ce 3.2031 | kl 0.024785 | 8.91s since last
step 880/2000 | loss 0.3762 | ce 2.9844 | kl 0.024862 | 8.91s since last
step 900/2000 | loss 0.4250 | ce 3.3750 | kl 0.024930 | 9.03s since last
[Diag] step 900/2000 | ppl(det) 24.79 | ppl(MC,5) 24.74 | tok-acc(det) 42.8% | ECE(det) 0.008 | ECE(MC,5) 0.006
step 920/2000 | loss 0.3488 | ce 2.7656 | kl 0.025017 | 20.26s since last
step 940/2000 | loss 0.4270 | ce 3.3906 | kl 0.025080 | 8.91s since last
step 960/2000 | loss 0.3996 | ce 3.1719 | kl 0.025182 | 8.91s since last
step 980/2000 | loss 0.4094 | ce 3.2500 | kl 0.025243 | 8.89s since last
step 1000/2000 | loss 0.3938 | ce 3.1250 | kl 0.025309 | 8.99s since last
[Diag] step 1000/2000 | ppl(det) 24.73 | ppl(MC,5) 24.67 | tok-acc(det) 42.9% | ECE(det) 0.010 | ECE(MC,5) 0.006
[Prune@1000] adapter params: 840416 -> 832734  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 24, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 22, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 24, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 30, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 1020/2000 | loss 0.3742 | ce 2.9688 | kl 0.025226 | 20.05s since last
step 1040/2000 | loss 0.4016 | ce 3.1875 | kl 0.025288 | 9.05s since last
step 1060/2000 | loss 0.4016 | ce 3.1875 | kl 0.025342 | 8.90s since last
step 1080/2000 | loss 0.3977 | ce 3.1562 | kl 0.025400 | 8.96s since last
step 1100/2000 | loss 0.3821 | ce 3.0312 | kl 0.025461 | 8.92s since last
[Diag] step 1100/2000 | ppl(det) 24.68 | ppl(MC,5) 24.63 | tok-acc(det) 42.9% | ECE(det) 0.011 | ECE(MC,5) 0.007
step 1120/2000 | loss 0.4055 | ce 3.2188 | kl 0.025513 | 20.04s since last
step 1140/2000 | loss 0.3821 | ce 3.0312 | kl 0.025548 | 9.08s since last
step 1160/2000 | loss 0.4055 | ce 3.2188 | kl 0.025588 | 8.92s since last
step 1180/2000 | loss 0.4309 | ce 3.4219 | kl 0.025632 | 8.93s since last
step 1200/2000 | loss 0.3919 | ce 3.1094 | kl 0.025683 | 9.06s since last
[Diag] step 1200/2000 | ppl(det) 24.63 | ppl(MC,5) 24.59 | tok-acc(det) 42.9% | ECE(det) 0.010 | ECE(MC,5) 0.008
[Prune@1200] adapter params: 832734 -> 805847  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 22, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 20, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 23, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 28, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 12, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 1220/2000 | loss 0.4367 | ce 3.4688 | kl 0.025228 | 20.02s since last
step 1240/2000 | loss 0.4075 | ce 3.2344 | kl 0.025315 | 8.88s since last
step 1260/2000 | loss 0.4290 | ce 3.4062 | kl 0.025358 | 8.91s since last
step 1280/2000 | loss 0.4133 | ce 3.2812 | kl 0.025395 | 9.07s since last
step 1300/2000 | loss 0.3547 | ce 2.8125 | kl 0.025433 | 8.94s since last
[Diag] step 1300/2000 | ppl(det) 24.60 | ppl(MC,5) 24.57 | tok-acc(det) 42.9% | ECE(det) 0.010 | ECE(MC,5) 0.007
step 1320/2000 | loss 0.4251 | ce 3.3750 | kl 0.025486 | 20.05s since last
step 1340/2000 | loss 0.4173 | ce 3.3125 | kl 0.025520 | 9.03s since last
step 1360/2000 | loss 0.4134 | ce 3.2812 | kl 0.025581 | 8.91s since last
step 1380/2000 | loss 0.3938 | ce 3.1250 | kl 0.025612 | 8.96s since last
step 1400/2000 | loss 0.4173 | ce 3.3125 | kl 0.025652 | 8.98s since last
[Diag] step 1400/2000 | ppl(det) 24.55 | ppl(MC,5) 24.51 | tok-acc(det) 42.9% | ECE(det) 0.011 | ECE(MC,5) 0.007
[Prune@1400] adapter params: 805847 -> 798933  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 21, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 20, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 23, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 28, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 11, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 1420/2000 | loss 0.3665 | ce 2.9062 | kl 0.025528 | 20.29s since last
step 1440/2000 | loss 0.3938 | ce 3.1250 | kl 0.025570 | 8.96s since last
step 1460/2000 | loss 0.4427 | ce 3.5156 | kl 0.025601 | 8.97s since last
step 1480/2000 | loss 0.3821 | ce 3.0312 | kl 0.025637 | 8.98s since last
step 1500/2000 | loss 0.4114 | ce 3.2656 | kl 0.025679 | 9.07s since last
[Diag] step 1500/2000 | ppl(det) 24.52 | ppl(MC,5) 24.50 | tok-acc(det) 42.9% | ECE(det) 0.012 | ECE(MC,5) 0.008
step 1520/2000 | loss 0.3958 | ce 3.1406 | kl 0.025710 | 20.23s since last
step 1540/2000 | loss 0.4017 | ce 3.1875 | kl 0.025749 | 8.98s since last
step 1560/2000 | loss 0.4056 | ce 3.2188 | kl 0.025769 | 8.96s since last
step 1580/2000 | loss 0.4056 | ce 3.2188 | kl 0.025801 | 8.96s since last
step 1600/2000 | loss 0.3958 | ce 3.1406 | kl 0.025813 | 8.94s since last
[Diag] step 1600/2000 | ppl(det) 24.50 | ppl(MC,5) 24.49 | tok-acc(det) 42.9% | ECE(det) 0.010 | ECE(MC,5) 0.007
[Prune@1600] adapter params: 798933 -> 795092  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 21, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 20, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 22, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 28, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 11, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 1620/2000 | loss 0.4212 | ce 3.3438 | kl 0.025753 | 20.07s since last
step 1640/2000 | loss 0.4095 | ce 3.2500 | kl 0.025773 | 9.29s since last
step 1660/2000 | loss 0.4271 | ce 3.3906 | kl 0.025804 | 8.97s since last
step 1680/2000 | loss 0.3978 | ce 3.1562 | kl 0.025821 | 8.92s since last
step 1700/2000 | loss 0.4036 | ce 3.2031 | kl 0.025841 | 8.94s since last
[Diag] step 1700/2000 | ppl(det) 24.52 | ppl(MC,5) 24.49 | tok-acc(det) 42.9% | ECE(det) 0.011 | ECE(MC,5) 0.008
step 1720/2000 | loss 0.3841 | ce 3.0469 | kl 0.025855 | 20.08s since last
step 1740/2000 | loss 0.4114 | ce 3.2656 | kl 0.025872 | 8.97s since last
step 1760/2000 | loss 0.4095 | ce 3.2500 | kl 0.025893 | 8.95s since last
step 1780/2000 | loss 0.4134 | ce 3.2812 | kl 0.025910 | 9.06s since last
step 1800/2000 | loss 0.3997 | ce 3.1719 | kl 0.025914 | 9.12s since last
[Diag] step 1800/2000 | ppl(det) 24.50 | ppl(MC,5) 24.46 | tok-acc(det) 42.9% | ECE(det) 0.010 | ECE(MC,5) 0.007
[Prune@1800] adapter params: 795092 -> 787410  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 14, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 21, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 20, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 21, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 27, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 11, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}
step 1820/2000 | loss 0.3938 | ce 3.1250 | kl 0.025759 | 20.01s since last
step 1840/2000 | loss 0.4368 | ce 3.4688 | kl 0.025778 | 8.96s since last
step 1860/2000 | loss 0.4114 | ce 3.2656 | kl 0.025785 | 8.96s since last
step 1880/2000 | loss 0.3802 | ce 3.0156 | kl 0.025794 | 9.02s since last
step 1900/2000 | loss 0.4036 | ce 3.2031 | kl 0.025801 | 8.98s since last
[Diag] step 1900/2000 | ppl(det) 24.50 | ppl(MC,5) 24.47 | tok-acc(det) 42.9% | ECE(det) 0.011 | ECE(MC,5) 0.008
step 1920/2000 | loss 0.4114 | ce 3.2656 | kl 0.025808 | 20.19s since last
step 1940/2000 | loss 0.4056 | ce 3.2188 | kl 0.025812 | 9.07s since last
step 1960/2000 | loss 0.3899 | ce 3.0938 | kl 0.025816 | 8.95s since last
step 1980/2000 | loss 0.4290 | ce 3.4062 | kl 0.025818 | 8.96s since last
step 2000/2000 | loss 0.4192 | ce 3.3281 | kl 0.025819 | 8.91s since last
[Diag] step 2000/2000 | ppl(det) 24.47 | ppl(MC,5) 24.48 | tok-acc(det) 43.0% | ECE(det) 0.010 | ECE(MC,5) 0.006
[Prune@2000] adapter params: 787410 -> 783569  ranks: {'transformer.h.0.attn.c_attn': 0, 'transformer.h.0.attn.c_proj': 1, 'transformer.h.0.mlp.c_fc': 0, 'transformer.h.0.mlp.c_proj': 0, 'transformer.h.1.attn.c_attn': 0, 'transformer.h.1.attn.c_proj': 0, 'transformer.h.1.mlp.c_fc': 2, 'transformer.h.1.mlp.c_proj': 0, 'transformer.h.2.attn.c_attn': 0, 'transformer.h.2.attn.c_proj': 0, 'transformer.h.2.mlp.c_fc': 9, 'transformer.h.2.mlp.c_proj': 0, 'transformer.h.3.attn.c_attn': 0, 'transformer.h.3.attn.c_proj': 1, 'transformer.h.3.mlp.c_fc': 13, 'transformer.h.3.mlp.c_proj': 0, 'transformer.h.4.attn.c_attn': 0, 'transformer.h.4.attn.c_proj': 0, 'transformer.h.4.mlp.c_fc': 12, 'transformer.h.4.mlp.c_proj': 0, 'transformer.h.5.attn.c_attn': 0, 'transformer.h.5.attn.c_proj': 1, 'transformer.h.5.mlp.c_fc': 14, 'transformer.h.5.mlp.c_proj': 0, 'transformer.h.6.attn.c_attn': 1, 'transformer.h.6.attn.c_proj': 0, 'transformer.h.6.mlp.c_fc': 15, 'transformer.h.6.mlp.c_proj': 0, 'transformer.h.7.attn.c_attn': 0, 'transformer.h.7.attn.c_proj': 0, 'transformer.h.7.mlp.c_fc': 21, 'transformer.h.7.mlp.c_proj': 0, 'transformer.h.8.attn.c_attn': 0, 'transformer.h.8.attn.c_proj': 0, 'transformer.h.8.mlp.c_fc': 20, 'transformer.h.8.mlp.c_proj': 0, 'transformer.h.9.attn.c_attn': 0, 'transformer.h.9.attn.c_proj': 0, 'transformer.h.9.mlp.c_fc': 21, 'transformer.h.9.mlp.c_proj': 0, 'transformer.h.10.attn.c_attn': 4, 'transformer.h.10.attn.c_proj': 0, 'transformer.h.10.mlp.c_fc': 27, 'transformer.h.10.mlp.c_proj': 0, 'transformer.h.11.attn.c_attn': 11, 'transformer.h.11.attn.c_proj': 0, 'transformer.h.11.mlp.c_fc': 32, 'transformer.h.11.mlp.c_proj': 4}

=== Results (WikiText-2) ===
Method: bayeslora
Perplexity (det): 24.500 | Perplexity (MC, 5): 24.461
Token-Acc (det): 41.00% | ECE(det): 0.0132
Token-Acc (MC):  41.01% | ECE(MC,5): 0.0097
```