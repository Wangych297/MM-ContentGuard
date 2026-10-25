"""
文件功能：图片模型推理器。加载训练好的 ImageRiskModel 检查点，
对输入图片执行推理并返回预测类别和置信度。

核心类：
- ImageModelPredictor: 图片风险模型推理器
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common.config import load_yaml, resolve_path
from common.image_io import load_rgb_image
from model.dataset import build_eval_transform
from model.labels import ID_TO_LABEL, IMAGE_LABELS, is_risk_label
from model.model import build_model_from_config


class ImageModelPredictor:
    def __init__(
        self,
        config_path: str | Path = "configs/image_classifier_dinov3_large.yaml",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = load_yaml(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        inference_cfg = self.config.get("inference", {})
        ckpt_path = resolve_path(
            checkpoint_path
            or inference_cfg.get(
                "checkpoint_path",
                "checkpoints/image_model_dinov3_large/best.pt",
            )
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model = build_model_from_config(self.config, num_classes=len(IMAGE_LABELS))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        image_size = int(self.config.get("model", {}).get("image_size", 224))
        self.transform = build_eval_transform(image_size)
        self.model_risk_threshold = float(inference_cfg.get("model_risk_threshold", 0.65))
        self.suspicious_model_threshold = float(inference_cfg.get("suspicious_model_threshold", 0.45))

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict[str, Any]:
        image = load_rgb_image(image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=-1)[0].detach().cpu()
        pred_id = int(probabilities.argmax().item())
        label = ID_TO_LABEL[pred_id]
        confidence = float(probabilities[pred_id].item())
        risk_probability = float(
            sum(probabilities[idx].item() for idx, label_name in ID_TO_LABEL.items() if is_risk_label(label_name))
        )
        if label == "safe":
            risk_level = "normal"
        elif confidence >= self.model_risk_threshold:
            risk_level = "risk"
        elif confidence >= self.suspicious_model_threshold:
            risk_level = "suspicious"
        else:
            risk_level = "normal"
        return {
            "model_label": label,
            "model_confidence": confidence,
            "model_risk_probability": risk_probability,
            "risk_level": risk_level,
            "probabilities": {
                ID_TO_LABEL[idx]: float(probabilities[idx].item()) for idx in range(len(ID_TO_LABEL))
            },
        }



