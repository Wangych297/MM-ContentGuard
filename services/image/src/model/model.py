"""
文件功能：图片风险分类模型。基于 torchvision 预训练骨干网络
（如 ConvNeXt、DINOv3）的图片风险分类模型，包含特征提取器
和自定义分类头。

核心类：
- TorchvisionFeatureExtractor: torchvision 特征提取器封装
- ImageRiskModel: 图片风险分类模型

核心函数：
- build_model_from_config: 根据 YAML 配置构建模型实例
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from common.config import resolve_path


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.num_features = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        return torch.flatten(features, 1)


class ImageRiskModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backend: str = "timm",
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        image_size: int = 224,
        dropout: float = 0.2,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.backbone_name = backbone_name
        self.backbone, feature_dim = self._create_backbone(
            backend, backbone_name, pretrained, image_size, checkpoint_path
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def _create_backbone(
        self,
        backend: str,
        backbone_name: str,
        pretrained: bool,
        image_size: int,
        checkpoint_path: str | None = None,
    ) -> tuple[nn.Module, int]:
        if backend == "timm":
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Install timm to use the default DINOv2 backbone.") from exc
            timm_kwargs: dict[str, Any] = {}
            if checkpoint_path:
                timm_kwargs["checkpoint_path"] = str(resolve_path(checkpoint_path))
            backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
                **timm_kwargs,
            )
            feature_dim = int(getattr(backbone, "num_features"))
            return backbone, feature_dim

        if backend == "torch_hub":
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=pretrained)
            feature_dim = int(getattr(backbone, "num_features", 384))
            return backbone, feature_dim

        if backend == "torchvision":
            try:
                from torchvision import models
            except ImportError as exc:
                raise ImportError("Install torchvision to use torchvision backbones.") from exc

            if backbone_name == "convnext_base":
                weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.convnext_base(weights=weights)
                return TorchvisionFeatureExtractor(model, 1024), 1024
            if backbone_name == "convnext_small":
                weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.convnext_small(weights=weights)
                return TorchvisionFeatureExtractor(model, 768), 768
            if backbone_name == "convnext_tiny":
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.convnext_tiny(weights=weights)
                return TorchvisionFeatureExtractor(model, 768), 768
            raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")

        raise ValueError(f"Unsupported backbone backend: {backend}")

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features: Any = self.backbone(images)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            elif "pooled" in features:
                features = features["pooled"]
            else:
                first_key = next(iter(features))
                features = features[first_key]
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def build_model_from_config(config: dict, num_classes: int) -> ImageRiskModel:
    model_cfg = config.get("model", {})
    return ImageRiskModel(
        num_classes=num_classes,
        backend=str(model_cfg.get("backend", "timm")),
        backbone_name=str(model_cfg.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        image_size=int(model_cfg.get("image_size", 224)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        checkpoint_path=model_cfg.get("checkpoint_path"),
    )

