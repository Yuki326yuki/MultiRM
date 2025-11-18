#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="multirm/config.yaml"

# 1) 生成训练数据（你可以只跑一次）
python -m multirm.prepare_multirm_data \
  --output data/multirm_train.jsonl \
  --max_ultra_bin 30000 \
  --max_ultra 10000 \
  --max_helpsteer2 5000

# 2) 训练多类型 Reward Model
python -u openrlhf/cli/train_multirm.py --config "$CONFIG"

# 3) 训练好之后，用 RewardBench 做评估
# 假设最终 ckpt 在 outputs/multirm-8b/final.pt
python -m multirm.eval_rewardbench \
  --ckpt outputs/multirm-8b/final.pt \
  --config "$CONFIG" \
  --type-name overall \
  --split train

