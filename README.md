# BayesLoRA

## Exp 1 (SST2+DistilBERT benchmark)

`python run_bayeslora_sst2.py --epochs 2 --r 8 --batch_size 32 --mc_T 8`

## Exp 2 (boolq+llama2 benchmark)

`python run_bayeslora_boolq_llama.py --model_name meta-llama/Llama-2-7b-hf --epochs 1 --mc_T 16`