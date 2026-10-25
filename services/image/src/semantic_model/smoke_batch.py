"""
文件功能：语义模型冒烟测试。在少量样本上快速运行推理，
验证 SemanticMemeModel 是否能正常加载和预测。

核心函数：
- main: 冒烟测试入口
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from common.config import load_yaml, resolve_path
from semantic_model.dataset import SemanticMemeDataset
from semantic_model.model import build_semantic_model_from_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_meme.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-backward", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.time()
    config = load_yaml(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    processor_name = model_cfg.get("processor_name") or model_cfg.get("name")

    print(f"config={args.config}", flush=True)
    print(f"model={model_cfg.get('name')}", flush=True)
    print(f"processor={processor_name}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    print("loading_dataset", flush=True)
    dataset = SemanticMemeDataset(
        resolve_path(data_cfg.get("train_file", "data/processed/semantic_meme/train.jsonl")),
        processor_name=str(processor_name),
        max_length=int(model_cfg.get("max_length", 64)),
        use_descriptions=bool(model_cfg.get("use_descriptions", True)),
    )
    sample_count = min(args.batch_size, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=sample_count,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    print(f"batch_loaded size={sample_count} pixel_values={tuple(batch['pixel_values'].shape)}", flush=True)

    print("loading_model", flush=True)
    model = build_semantic_model_from_config(config).to(device)
    model.train()
    print(f"model_loaded hidden_dim={model.hidden_dim}", flush=True)

    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)

    print("forward", flush=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            text_description_input_ids=batch["text_description_input_ids"],
            text_description_attention_mask=batch["text_description_attention_mask"],
            meme_description_input_ids=batch["meme_description_input_ids"],
            meme_description_attention_mask=batch["meme_description_attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        loss = (
            nn.CrossEntropyLoss()(outputs["type_logits"], batch["type_label"])
            + 0.5 * nn.CrossEntropyLoss()(outputs["binary_logits"], batch["binary_label"])
            + 0.2 * nn.BCEWithLogitsLoss()(outputs["modal_logits"], batch["modal_label"])
        )
    print(f"forward_ok loss={float(loss.detach().cpu()):.6f}", flush=True)

    if not args.no_backward:
        print("backward", flush=True)
        loss.backward()
        optimizer.step()
        print("backward_ok", flush=True)

    if torch.cuda.is_available():
        print(
            f"max_cuda_memory_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}",
            flush=True,
        )
    print(f"elapsed_sec={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
