"""
文件功能：语义 Meme 数据集。基于 ChineseCLIP 的语义分类 Dataset，
同时处理图片和文本两种模态，用于梗图/表情包等图文混合内容的
语义风险分类。

核心类：
- SemanticMemeDataset: 语义 Meme 分类 PyTorch Dataset

核心函数：
- load_processor: 加载 ChineseCLIP 处理器
- process_image / tokenize_text: 图片和文本的预处理
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ChineseCLIPImageProcessor, ChineseCLIPProcessor

from common.image_io import load_rgb_image
from data.io import read_jsonl


def load_processor(processor_name: str) -> ChineseCLIPProcessor:
    try:
        return ChineseCLIPProcessor.from_pretrained(processor_name)
    except TypeError as exc:
        if "multiple values for argument 'image_processor'" not in str(exc):
            raise
        image_processor = ChineseCLIPImageProcessor.from_pretrained(processor_name)
        tokenizer = AutoTokenizer.from_pretrained(processor_name)
        return ChineseCLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)


def tokenize_text(processor: ChineseCLIPProcessor, text: str, max_length: int) -> dict[str, torch.Tensor]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    else:
        encoded = processor(
            text=text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    return {key: value.squeeze(0) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}


def process_image(processor: ChineseCLIPProcessor, image_path: str | Path) -> torch.Tensor:
    image = load_rgb_image(image_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = getattr(processor, "feature_extractor", None)
    if image_processor is not None:
        encoded = image_processor(images=image, return_tensors="pt")
    else:
        encoded = processor(images=image, return_tensors="pt")
    return encoded["pixel_values"].squeeze(0)


class SemanticMemeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        processor_name: str,
        max_length: int = 64,
        use_descriptions: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = list(read_jsonl(self.manifest_path))
        if not self.records:
            raise ValueError(f"Manifest split is empty: {self.manifest_path}")
        self.processor = load_processor(processor_name)
        self.max_length = max_length
        self.use_descriptions = use_descriptions

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        text = str(record.get("text") or "")
        text_description = str(record.get("text_description") or "") if self.use_descriptions else ""
        meme_description = str(record.get("meme_description") or "") if self.use_descriptions else ""

        text_inputs = tokenize_text(self.processor, text, self.max_length)
        text_description_inputs = tokenize_text(self.processor, text_description, self.max_length)
        meme_description_inputs = tokenize_text(self.processor, meme_description, self.max_length)
        pixel_values = process_image(self.processor, record["image_path"])

        return {
            "text_input_ids": text_inputs["input_ids"],
            "text_attention_mask": text_inputs["attention_mask"],
            "text_description_input_ids": text_description_inputs["input_ids"],
            "text_description_attention_mask": text_description_inputs["attention_mask"],
            "meme_description_input_ids": meme_description_inputs["input_ids"],
            "meme_description_attention_mask": meme_description_inputs["attention_mask"],
            "pixel_values": pixel_values,
            "binary_label": torch.tensor(int(record["binary_label"]), dtype=torch.long),
            "type_label": torch.tensor(int(record["type_label"]), dtype=torch.long),
            "modal_label": torch.tensor(
                [int(record.get("text_modal", 0)), int(record.get("image_modal", 0))],
                dtype=torch.float32,
            ),
            "id": str(record.get("id") or ""),
            "image_path": str(record["image_path"]),
            "text": text,
            "text_description": text_description,
            "meme_description": meme_description,
        }

    @property
    def type_labels(self) -> list[int]:
        return [int(record["type_label"]) for record in self.records]

    @property
    def binary_labels(self) -> list[int]:
        return [int(record["binary_label"]) for record in self.records]
