"""
文件功能：语义 Meme 分类模型。基于 ChineseCLIP 的图文多模态
分类模型，同时处理图片和文本输入，输出语义类型、二分类和
模态三个维度的预测结果。

核心类：
- SemanticMemeModel: 语义 Meme 多模态分类模型

核心函数：
- build_semantic_model_from_config: 根据 YAML 配置构建模型实例
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import ChineseCLIPModel

from semantic_model.labels import BINARY_LABELS, MODAL_LABELS, SEMANTIC_TYPE_LABELS


class SemanticMemeModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_descriptions: bool = True,
        description_dropout: float = 0.2,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.clip = ChineseCLIPModel.from_pretrained(model_name)
        self.use_descriptions = use_descriptions
        self.description_dropout = description_dropout
        hidden_dim = self._projection_dim()
        self.hidden_dim = hidden_dim

        self.description_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.type_head = nn.Linear(fusion_hidden_dim, len(SEMANTIC_TYPE_LABELS))
        self.binary_head = nn.Linear(fusion_hidden_dim, len(BINARY_LABELS))
        self.modal_head = nn.Linear(fusion_hidden_dim, len(MODAL_LABELS))

    def _projection_dim(self) -> int:
        for attr in ("projection_dim", "hidden_size"):
            value = getattr(self.clip.config, attr, None)
            if value is not None:
                return int(value)
        text_config = getattr(self.clip.config, "text_config", None)
        if text_config is not None:
            for attr in ("projection_dim", "hidden_size"):
                value = getattr(text_config, attr, None)
                if value is not None:
                    return int(value)
        return 512

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.clip.parameters():
            parameter.requires_grad = trainable

    def _maybe_drop_descriptions(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.training or self.description_dropout <= 0:
            return feature
        keep = torch.empty(feature.shape[0], 1, device=feature.device).bernoulli_(
            1.0 - self.description_dropout
        )
        return feature * keep

    @staticmethod
    def _feature_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if torch.is_tensor(pooler_output):
            return pooler_output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.ndim == 2:
                    return item
        raise TypeError(f"Cannot extract feature tensor from {type(output)!r}")

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_description_input_ids: torch.Tensor,
        text_description_attention_mask: torch.Tensor,
        meme_description_input_ids: torch.Tensor,
        meme_description_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        use_descriptions: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        text_feature = self._feature_tensor(
            self.clip.get_text_features(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
        )
        image_feature = self._feature_tensor(
            self.clip.get_image_features(pixel_values=pixel_values)
        )

        text_feature = F.normalize(text_feature, dim=-1)
        image_feature = F.normalize(image_feature, dim=-1)
        descriptions_enabled = self.use_descriptions if use_descriptions is None else use_descriptions

        if descriptions_enabled:
            text_description_feature = self._feature_tensor(
                self.clip.get_text_features(
                    input_ids=text_description_input_ids,
                    attention_mask=text_description_attention_mask,
                )
            )
            meme_description_feature = self._feature_tensor(
                self.clip.get_text_features(
                    input_ids=meme_description_input_ids,
                    attention_mask=meme_description_attention_mask,
                )
            )
            text_description_feature = self._maybe_drop_descriptions(
                F.normalize(text_description_feature, dim=-1)
            )
            meme_description_feature = self._maybe_drop_descriptions(
                F.normalize(meme_description_feature, dim=-1)
            )
            gate = torch.sigmoid(
                self.description_gate(
                    torch.cat(
                        [text_feature, text_description_feature, meme_description_feature],
                        dim=-1,
                    )
                )
            )
            enhanced_text = self.text_norm(
                text_feature
                + gate[:, 0:1] * text_description_feature
                + gate[:, 1:2] * meme_description_feature
            )
        else:
            gate = torch.zeros(text_feature.shape[0], 2, device=text_feature.device)
            enhanced_text = self.text_norm(text_feature)

        fused = torch.cat(
            [
                image_feature,
                enhanced_text,
                image_feature * enhanced_text,
                torch.abs(image_feature - enhanced_text),
            ],
            dim=-1,
        )
        hidden = self.fusion(fused)
        return {
            "type_logits": self.type_head(hidden),
            "binary_logits": self.binary_head(hidden),
            "modal_logits": self.modal_head(hidden),
            "description_gate": gate,
        }


def build_semantic_model_from_config(config: dict[str, Any]) -> SemanticMemeModel:
    model_cfg = config.get("model", {})
    return SemanticMemeModel(
        model_name=str(model_cfg.get("name", "OFA-Sys/chinese-clip-vit-large-patch14")),
        use_descriptions=bool(model_cfg.get("use_descriptions", True)),
        description_dropout=float(model_cfg.get("description_dropout", 0.2)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
