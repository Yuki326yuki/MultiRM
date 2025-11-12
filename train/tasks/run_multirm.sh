#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="multirm/config.example.yaml"

# 单卡（LoRA，先小步验证）
python -u openrlhf/cli/train_multirm.py --config "$CONFIG"

# 多卡 DeepSpeed（如需要，取消注释并配置你的GPU列表）
# deepspeed openrlhf/cli/train_multirm.py --config "$CONFIG" --deepspeed multirm/ds_config_zero3.json
