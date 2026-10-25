"""
文件功能：PyTorch 数据集。封装文本风险样本的 PyTorch Dataset 和
批处理 collator，负责文本 tokenization、标签张量构建和批内 padding。

核心类：
- RiskTextDataset: 风险文本 PyTorch Dataset
- RiskBatchCollator: 动态批处理 collator，处理不定长序列
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from risk_text_detection.models.phonetic_glyph import PhoneticGlyphFeatureExtractor
from risk_text_detection.models.pinyin import PinyinCharTokenizer, text_to_pinyin
from risk_text_detection.training.labeling import LabelVocab, read_jsonl


class JsonlRiskDataset(Dataset):
    def __init__(self, path, label_vocab: LabelVocab, limit: int | None = None) -> None:
        self.records = read_jsonl(path, limit=limit)
        self.label_vocab = label_vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "id": record["id"],
            "text": record["text"],
            "labels": self.label_vocab.encode_record(record),
            "metadata": record.get("metadata", {}),
        }


@dataclass
class RiskBatchCollator:
    tokenizer: PreTrainedTokenizerBase
    label_vocab: LabelVocab
    max_length: int = 128
    use_pinyin: bool = False
    pinyin_max_length: int = 256
    use_phonetic_glyph: bool = False
    phonetic_glyph_index_file: str = "data/raw/ast/hanzi.txt"
    phonetic_glyph_neighbor_top_k: int = 4
    augmenter: Any | None = None

    def __post_init__(self) -> None:
        self.pinyin_tokenizer = PinyinCharTokenizer() if self.use_pinyin else None
        self.phonetic_glyph_extractor = (
            PhoneticGlyphFeatureExtractor(
                index_file=self.phonetic_glyph_index_file,
                neighbor_top_k=self.phonetic_glyph_neighbor_top_k,
            )
            if self.use_phonetic_glyph
            else None
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        if self.augmenter is not None:
            texts = [self.augmenter.augment(text) for text in texts]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=self.use_phonetic_glyph,
        )
        offset_mapping = encoded.pop("offset_mapping", None)

        labels = {
            task: torch.tensor([item["labels"][task] for item in batch], dtype=torch.long)
            for task in self.label_vocab.tasks
        }

        result: dict[str, Any] = {
            **encoded,
            "labels": labels,
            "ids": [item["id"] for item in batch],
            "texts": texts,
        }

        if self.use_pinyin and self.pinyin_tokenizer is not None:
            pinyin_texts = [text_to_pinyin(text) for text in texts]
            result["pinyin_ids"] = self.pinyin_tokenizer.batch_encode(
                pinyin_texts,
                max_length=self.pinyin_max_length,
            )

        if self.use_phonetic_glyph and self.phonetic_glyph_extractor is not None:
            if offset_mapping is None:
                raise ValueError("offset_mapping is required when use_phonetic_glyph=True")
            result["phonetic_glyph"] = self.phonetic_glyph_extractor.batch_encode(
                texts,
                offset_mapping,
            )

        return result
