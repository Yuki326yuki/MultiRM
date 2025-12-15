#!/usr/bin/env bash

#data 
#V1
CUDA_VISIBLE_DEVICES=7 python -m multirm.prepare_gsm8k_implicitprm_V1 \
  --train-file data/gsm8k_train_raw.jsonl \
  --test-file  data/gsm8k_test_raw.jsonl \
  --output-dir data \
  --llm-model  /hpc2hdd/home/jianmu/home/models/Qwen2.5-7B-Instruct\
  --device cuda \
  --max-new-tokens 256 \
  --max-llm-neg 2


#V3 Batch
CUDA_VISIBLE_DEVICES=7 python -m multirm.prepare_gsm8k_implicitprm_V2 \
  --train-file data/gsm8k_train_raw.jsonl \
  --test-file  data/gsm8k_test_raw.jsonl \
  --output-dir data \
  --llm-model  /hpc2hdd/home/jianmu/home/models/Qwen2.5-7B-Instruct \
  --device cuda \
  --max-new-tokens 128 \
  --max-llm-neg 2 \
  --types math_final_answer,format_adherence \
  --no-wrong-final \
  --gen-batch-size 8 \
  --neg-oversample 12 \
  --neg-max-rounds 2 \

#train token-level
CUDA_VISIBLE_DEVICES=7 python -m openrlhf.cli.train_multirmt \
  --config multirm/config.gsm8k.yaml


#test rmt score and base
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_multirmt_score \
  --scorer multirm \
  --rm-ckpt outputs/gsm8k-multirmt-0.5b_llm_2/final.pt \
  --test-file data/gsm8k_test_multirm_llm.jsonl \
  --rm-type math_final_answer \
  --out-path evaluation/eval_gsm8k_rmt/eval_gsm8k_rmt_score/gsm8k_rmt_socre_2.3.jsonl \
  --max-samples 200

CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_multirmt_score \
  --scorer logprob \
  --lm-model /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct \
  --test-file data/gsm8k_test_multirm_llm.jsonl \
  --rm-type math_final_answer \
  --out-path evaluation/eval_gsm8k_rmt/eval_gsm8k_rmt_base_score/gsm8k_rmt_base_score_1.1.jsonl


#test token-level
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_rmt \
  --gsm8k_path data/gsm8k_test_raw.jsonl \
  --model_type multirm \
  --model_path outputs/gsm8k-multirmt-0.5b_llm_2/final.pt \
  --out-dir evaluation/eval_gsm8k_rmt/eval_gsm8k_rmt_mrm \
  --limit 100


CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_gsm8k_rmt \
  --gsm8k_path data/gsm8k_test_raw.jsonl \
  --model_type hf \
  --model_path /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct \
  --out-dir evaluation/eval_gsm8k_rmt/eval_gsm8k_rmt_base\
  --limit 100


#test by opencompass
CUDA_VISIBLE_DEVICES=7  opencompass --models hf_qwen2_5_0_5b_rm --datasets demo_gsm8k_chat_gen
CUDA_VISIBLE_DEVICES=7  opencompass --models hf_qwen2_5_0_5b_instruct --datasets gsm8k_gen
CUDA_VISIBLE_DEVICES=7  opencompass --models hf_qwen2_5_0_5b_rm --datasets gsm8k_gen

#导出 for opencompass
CUDA_VISIBLE_DEVICES=7 python -m multirm.export_rm_to_hf
