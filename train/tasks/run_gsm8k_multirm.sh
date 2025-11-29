#!/usr/bin/env bash

#data
CUDA_VISIBLE_DEVICES=7 python -m multirm.prepare_gsm8k_implicitprm_llm \
  --train-file data/gsm8k_train_raw.jsonl \
  --test-file  data/gsm8k_test_raw.jsonl \
  --output-dir data \
  --llm-model  /hpc2hdd/home/jianmu/home/models/Qwen2.5-7B-Instruct\
  --device cuda \
  --max-new-tokens 256 \
  --max-llm-neg 2

# train
CUDA_VISIBLE_DEVICES=7 python -m openrlhf.cli.train_multirm \
  --config multirm/config.gsm8k.yaml


# test
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_rm \
  --ckpt outputs/gsm8k-multirm-0.5b/final.pt \
  --config multirm/config.gsm8k.yaml \
  --data data/gsm8k_test_multirm.jsonl \
  --type-name math_correctness

# basemodel
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_basemodel \
  --model /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_test_multirm_llm.jsonl \
  --type-name math_correctness

#llm版本
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_rm_llm \
  --ckpt outputs/gsm8k-multirm-0.5b_llm/final.pt \
  --data-file data/gsm8k_test_multirm_llm.jsonl \
  --out-dir eval/eval_gsm8k_llm \
  --device cuda \
  --max-len 2048

#llm版本basemodel
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_rm_llm_basemodel \
  --model /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct \
  --data-file data/gsm8k_test_multirm_llm.jsonl \
  --out-dir eval/eval_gsm8k_llm  \
  --device cuda