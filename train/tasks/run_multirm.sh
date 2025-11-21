#!/usr/bin/env bash
set -euo pipefail

# 1. 切换到项目目录
cd /hpc2hdd/home/jianmu/home/chenyu_multiRM/MultiRM/train

# 2. 只使用 GPU 7（A800-80G）
export CUDA_VISIBLE_DEVICES=7
echo "[INFO] Using GPU: $CUDA_VISIBLE_DEVICES"

# 3. 把 train 目录加入 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[INFO] PYTHONPATH = $PYTHONPATH"

# 4. 生成多类型 RM 训练数据
#echo "[INFO] Generating unified MultiType-RM training data..."
#python -m multirm.prepare_multirm_data \
#    --output data/multirm_train.jsonl \
#    --max_ultra_bin 30000 \
#    --max_ultra 10000 \
#    --max_helpsteer2 5000

# 5. 训练多类型 Reward Model
echo "[INFO] Training MultiType Reward Model..."
python -m openrlhf.cli.train_multirm \
    --config multirm/config.example.yaml

# 6. RewardBench 评估
echo "[INFO] Running RewardBench evaluation..."
python -m multirm.eval_rewardbench \
    --ckpt outputs/multirm-Skywork-Reward-V2-Llama-3.2-1B/final.pt \
    --config multirm/config.example.yaml \
    --type-name overall \
    --local-data data/reward_bench_filtered.jsonl\
    --split train

echo "[DONE] All steps finished."

#打分baseline
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_rewardbench_basemodel \
  --model /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct \
  --local-data data/reward_bench_filtered.jsonl


#生成baseline
CUDA_VISIBLE_DEVICES=7 python -m multirm.eval_rewardbench_generative \
  --model /hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct  \
  --local-data data/reward_bench_filtered.jsonl \
  --max-new-tokens 16 \
  --temperature 0.0