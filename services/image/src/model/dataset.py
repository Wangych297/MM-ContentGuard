"""
文件功能：图片分类数据集。封装图片风险分类的 PyTorch Dataset，
提供训练和评估两种数据增强变换策略。

核心类：
- ImageDataset: 图片分类 Dataset，支持从清单文件加载

核心函数：
- build_train_transform: 构建训练数据增强流水线
- build_eval_transform: 构建评估数据预处理流水线
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common.image_io import load_rgb_image
from data.io import read_jsonl
from model.labels import LABEL_TO_ID


def build_train_transform(image_size: int, augmentation_cfg: dict | None = None):
    augmentation_cfg = augmentation_cfg or {}
    crop_scale = tuple(augmentation_cfg.get("random_resized_crop_scale", (0.65, 1.0)))
    color_jitter = tuple(augmentation_cfg.get("color_jitter", (0.15, 0.15, 0.12, 0.02)))

    steps = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=float(augmentation_cfg.get("hflip", 0.5))),
    ]
    if augmentation_cfg.get("randaugment", False):
        steps.append(
            transforms.RandAugment(
                num_ops=int(augmentation_cfg.get("randaugment_num_ops", 2)),
                magnitude=int(augmentation_cfg.get("randaugment_magnitude", 7)),
            )
        )
    steps.extend(
        [
            transforms.ColorJitter(
                brightness=float(color_jitter[0]),
                contrast=float(color_jitter[1]),
                saturation=float(color_jitter[2]),
                hue=float(color_jitter[3]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    random_erasing = float(augmentation_cfg.get("random_erasing", 0.0))
    if random_erasing > 0:
        steps.append(transforms.RandomErasing(p=random_erasing, value="random"))
    return transforms.Compose(steps)


def build_eval_transform(image_size: int):
    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class ImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, transform=None) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = list(read_jsonl(self.manifest_path))
        if not self.records:
            raise ValueError(f"Manifest split is empty: {self.manifest_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = load_rgb_image(record["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        label = LABEL_TO_ID[record["label"]]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "id": record["id"],
            "image_path": record["image_path"],
        }

    @property
    def labels(self) -> list[int]:
        return [LABEL_TO_ID[record["label"]] for record in self.records]


