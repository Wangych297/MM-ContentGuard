"""
文件功能：语义模型评测。加载训练好的 SemanticMemeModel 检查点，
在测试集上计算多标签分类和模态分类指标。

核心函数：
- evaluate_checkpoint: 加载检查点并执行评测
- main: 命令行入口
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from common.config import load_yaml, resolve_path
from semantic_model.dataset import SemanticMemeDataset
from semantic_model.model import build_semantic_model_from_config
from semantic_model.train import evaluate


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | None = None,
    split: str = "test",
) -> dict:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    inference_cfg = config.get("inference", {})

    ckpt_path = resolve_path(
        checkpoint_path
        or inference_cfg.get("checkpoint_path", "checkpoints/semantic_model_focal_binary/best.pt")
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_semantic_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    split_file = data_cfg.get(f"{split}_file")
    if split_file is None:
        raise ValueError(f"Unknown semantic split: {split}")
    processor_name = str(
        model_cfg.get("processor_name") or model_cfg.get("name", "OFA-Sys/chinese-clip-vit-large-patch14")
    )
    dataset = SemanticMemeDataset(
        resolve_path(split_file),
        processor_name=processor_name,
        max_length=int(model_cfg.get("max_length", 64)),
        use_descriptions=bool(model_cfg.get("use_descriptions", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    metrics = evaluate(model, loader, device, train_cfg)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["split"] = split
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_meme_focal_binary.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = evaluate_checkpoint(load_yaml(args.config), args.checkpoint, args.split)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
