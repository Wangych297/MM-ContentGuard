"""
文件功能：模型冒烟测试。使用小型虚拟主干网络验证多任务模型的
前向传播是否正常运行而不报错。
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from risk_text_detection.models import multitask
from risk_text_detection.models.multitask import MacBertMultiTaskClassifier, MultiTaskModelConfig


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def test_multitask_forward_smoke(monkeypatch) -> None:
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
    model = MacBertMultiTaskClassifier(
        MultiTaskModelConfig(
            model_name="tiny",
            task_num_labels={"risk_binary": 2, "primary_category": 3},
            task_weights={"risk_binary": 1.0, "primary_category": 1.0},
        )
    )
    logits = model(
        input_ids=torch.ones((2, 5), dtype=torch.long),
        attention_mask=torch.ones((2, 5), dtype=torch.long),
    )

    assert logits["risk_binary"].shape == (2, 2)
    assert logits["primary_category"].shape == (2, 3)
