# -*- coding: utf-8 -*-
import argparse, os, json

import torch

from openrlhf.trainer.multirm_trainer import MultiTypeRMTrainer


def parse_config(path: str):
    if path.endswith(".json"):
        return json.load(open(path, "r", encoding="utf-8"))
    else:
        import yaml
        return yaml.safe_load(open(path, "r", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML/JSON config for MultiType RM")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    # 固定随机种子
    seed = cfg.get("seed", 42)
    import random
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    trainer = MultiTypeRMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
