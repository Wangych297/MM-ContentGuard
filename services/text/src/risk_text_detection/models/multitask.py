"""
文件功能：多任务文本审核模型。基于 MacBERT 主干，支持多标签分类
（风险二分类、主类别、子类别），可选加载拼音分支和音形融合分支
作为辅助输入特征。

核心类：
- MultiTaskModelConfig: 模型配置，包含模型路径、任务标签、损失权重等
- MacBertMultiTaskClassifier: 多任务分类器，MacBERT + 多任务分类头
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from risk_text_detection.models.phonetic_glyph import PhoneticGlyphBranch
from risk_text_detection.models.pinyin import PinyinBranch
from risk_text_detection.training.labeling import IGNORE_INDEX


@dataclass
class MultiTaskModelConfig:
    model_name: str = "models/hfl/chinese-macbert-base"
    task_num_labels: dict[str, int] = field(default_factory=dict)
    task_weights: dict[str, float] = field(default_factory=dict)
    dropout: float = 0.1
    use_safetensors: bool | None = False
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    use_pinyin: bool = False
    pinyin_hidden_dim: int = 256
    use_phonetic_glyph: bool = False
    phonetic_glyph_hidden_dim: int = 256
    phonetic_glyph_embed_dim: int = 48


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets.ne(self.ignore_index)
        if not mask.any():
            return logits.sum() * 0.0
        logits = logits[mask]
        targets = targets[mask]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class TaskHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        mid = max(hidden_size // 2, num_labels)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MacBertMultiTaskClassifier(nn.Module):
    def __init__(self, config: MultiTaskModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_safetensors is False:
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        backbone_config = AutoConfig.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(
            config.model_name,
            config=backbone_config,
            use_safetensors=config.use_safetensors,
        )
        hidden_size = int(backbone_config.hidden_size)

        self.phonetic_glyph_branch = None
        self.phonetic_glyph_projection = None
        self.phonetic_glyph_gate = None
        self.sequence_pooler = None
        if config.use_phonetic_glyph:
            self.phonetic_glyph_branch = PhoneticGlyphBranch(
                embed_dim=config.phonetic_glyph_embed_dim,
                hidden_dim=config.phonetic_glyph_hidden_dim,
                dropout=config.dropout,
            )
            self.phonetic_glyph_projection = nn.Linear(
                config.phonetic_glyph_hidden_dim,
                hidden_size,
            )
            self.phonetic_glyph_gate = nn.Linear(hidden_size * 2, hidden_size)
            self.sequence_pooler = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        self.pinyin_branch = None
        feature_size = hidden_size
        if config.use_pinyin:
            self.pinyin_branch = PinyinBranch(hidden_dim=config.pinyin_hidden_dim)
            feature_size += config.pinyin_hidden_dim

        self.fusion = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.heads = nn.ModuleDict(
            {
                task: TaskHead(hidden_size, num_labels, config.dropout)
                for task, num_labels in config.task_num_labels.items()
            }
        )
        self.focal_loss = FocalLoss(gamma=config.focal_gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        pinyin_ids: torch.Tensor | None = None,
        phonetic_glyph: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**inputs)
        token_features = outputs.last_hidden_state

        if self.phonetic_glyph_branch is not None:
            if phonetic_glyph is None:
                raise ValueError("phonetic_glyph is required when use_phonetic_glyph=True")
            pg_inputs = {
                key: value.to(token_features.device)
                for key, value in phonetic_glyph.items()
            }
            pg_features = self.phonetic_glyph_branch(pg_inputs)
            pg_projected = self.phonetic_glyph_projection(pg_features)
            gate = torch.sigmoid(
                self.phonetic_glyph_gate(torch.cat([token_features, pg_projected], dim=-1))
            )
            token_features = token_features + gate * pg_projected

            mask = attention_mask.unsqueeze(-1).to(token_features.dtype)
            mean_features = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            features = self.sequence_pooler(
                torch.cat([token_features[:, 0, :], mean_features], dim=-1)
            )
        else:
            features = token_features[:, 0, :]

        if self.pinyin_branch is not None:
            if pinyin_ids is None:
                raise ValueError("pinyin_ids is required when use_pinyin=True")
            pinyin_features = self.pinyin_branch(pinyin_ids.to(features.device))
            features = torch.cat([features, pinyin_features], dim=-1)

        fused = self.fusion(features)
        return {task: head(fused) for task, head in self.heads.items()}

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        for task, task_logits in logits.items():
            if task not in labels:
                continue
            task_labels = labels[task].to(task_logits.device)
            valid = task_labels.ne(IGNORE_INDEX)
            if not valid.any():
                continue
            if self.config.use_focal_loss:
                loss = self.focal_loss(task_logits, task_labels)
            else:
                loss = F.cross_entropy(
                    task_logits,
                    task_labels,
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=self.config.label_smoothing,
                )
            weight = float(self.config.task_weights.get(task, 1.0))
            losses[task] = loss * weight

        total = sum(losses.values()) if losses else next(iter(logits.values())).sum() * 0.0
        losses["total"] = total
        return total, losses
