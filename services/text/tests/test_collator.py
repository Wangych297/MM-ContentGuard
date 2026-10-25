"""
文件功能：批处理 Collator 单元测试。验证 RiskBatchCollator 是否正确构建
输入张量和标签矩阵。
"""

import torch

from risk_text_detection.data.torch_dataset import RiskBatchCollator
from risk_text_detection.training.labeling import LabelVocab


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        max_len = max(len(text) for text in texts) + 2
        return {
            "input_ids": torch.ones((len(texts), max_len), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), max_len), dtype=torch.long),
            "token_type_ids": torch.zeros((len(texts), max_len), dtype=torch.long),
        }


def test_collator_builds_tensors_and_labels() -> None:
    vocab = LabelVocab.build(
        [{"labels": {"risk_binary": 1, "primary_category": "spam_general"}}],
        ["risk_binary", "primary_category"],
    )
    collator = RiskBatchCollator(tokenizer=DummyTokenizer(), label_vocab=vocab, max_length=16)
    batch = collator(
        [
            {
                "id": "a",
                "text": "加我微信",
                "labels": {
                    "risk_binary": vocab.tasks["risk_binary"].label_to_id["1"],
                    "primary_category": vocab.tasks["primary_category"].label_to_id["spam_general"],
                },
                "metadata": {},
            }
        ]
    )

    assert batch["input_ids"].shape[0] == 1
    assert batch["labels"]["risk_binary"].item() == 1
    assert batch["ids"] == ["a"]
