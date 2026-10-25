"""
文件功能：图片模型评测。加载训练好的图片分类模型检查点，
在测试集上计算分类指标并输出评测结果。

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
from model.dataset import ImageDataset, build_eval_transform
from model.labels import IMAGE_LABELS
from model.model import build_model_from_config
from model.train import evaluate


def evaluate_checkpoint(config: dict, checkpoint_path: str | None = None) -> dict:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    inference_cfg = config.get("inference", {})
    ckpt_path = resolve_path(
        checkpoint_path
        or inference_cfg.get(
            "checkpoint_path",
            "checkpoints/image_model_dinov3_large/best.pt",
        )
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config, num_classes=len(IMAGE_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = ImageDataset(
        resolve_path(data_cfg.get("test_file", "data/processed/splits/image_test.jsonl")),
        transform=build_eval_transform(int(model_cfg.get("image_size", 224))),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    return evaluate(model, test_loader, device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_classifier_dinov3_large.yaml")
    parser.add_argument("--checkpoint")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    metrics = evaluate_checkpoint(config, args.checkpoint)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



