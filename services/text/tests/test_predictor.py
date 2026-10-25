"""
文件功能：推理器单元测试。验证 RiskTextPredictor 能否正确加载
模型检查点并对测试文本给出有效预测结果。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from risk_text_detection.inference import predictor as predictor_module
from risk_text_detection.inference.predictor import RiskTextPredictor
from risk_text_detection.models import multitask
from risk_text_detection.models.multitask import MacBertMultiTaskClassifier, MultiTaskModelConfig
from risk_text_detection.training.labeling import LabelVocab


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        max_len = max(len(text) for text in texts) + 2
        return {
            "input_ids": torch.ones((len(texts), max_len), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), max_len), dtype=torch.long),
            "token_type_ids": torch.zeros((len(texts), max_len), dtype=torch.long),
        }


def _write_tiny_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer").mkdir(exist_ok=True)
    vocab = LabelVocab.build(
        [
            {
                "labels": {
                    "risk_binary": 1,
                    "primary_category": "spam_general",
                    "toxic_binary": 0,
                }
            }
        ],
        ["risk_binary", "primary_category", "toxic_binary"],
    )
    vocab.save(path / "label_vocab.json")
    config = MultiTaskModelConfig(
        model_name="tiny",
        task_num_labels=vocab.num_labels,
        task_weights={task: 1.0 for task in vocab.tasks},
    )
    (path / "model_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False),
        encoding="utf-8",
    )
    model = MacBertMultiTaskClassifier(config)
    torch.save(model.state_dict(), path / "pytorch_model.bin")


def test_predictor_loads_checkpoint_and_predicts(monkeypatch) -> None:
    monkeypatch.setattr(
        multitask.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(hidden_size=16),
    )
    monkeypatch.setattr(
        multitask.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: TinyBackbone(hidden_size=16),
    )
    monkeypatch.setattr(
        predictor_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    checkpoint = Path("tmp") / "pytest_cases" / "predictor_checkpoint"
    _write_tiny_checkpoint(checkpoint)

    predictor = RiskTextPredictor(checkpoint_dir=checkpoint, device="cpu")
    result = predictor.predict("加我微信赚钱")

    assert result["risk_binary"] in {"safe", "suspicious", "risk"}
    assert result["primary_category"] is not None
    assert "probabilities" in result
    assert "risk_binary" in result["probabilities"]
