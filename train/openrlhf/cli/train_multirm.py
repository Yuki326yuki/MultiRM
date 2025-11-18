# train/openrlhf/cli/train_multirm.py
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import torch
import yaml

from openrlhf.trainer.multirm_trainer import MultiTypeRMTrainer


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="config yaml/json for MultiType RM")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = MultiTypeRMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
