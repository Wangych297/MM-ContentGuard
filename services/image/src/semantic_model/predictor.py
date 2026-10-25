"""
文件功能：语义模型推理器。加载训练好的 SemanticMemeModel 检查点，
对输入的图片-文本对执行推理，返回多维度预测结果。

核心类：
- SemanticMemePredictor: 语义 Meme 模型推理器
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common.config import load_yaml, resolve_path
from semantic_model.dataset import load_processor, process_image, tokenize_text
from semantic_model.description_provider import build_description_provider
from semantic_model.labels import BINARY_ID_TO_LABEL, MODAL_LABELS, TYPE_ID_TO_LABEL
from semantic_model.model import build_semantic_model_from_config


class SemanticMemePredictor:
    def __init__(
        self,
        config_path: str | Path = "configs/semantic_meme_focal_binary.yaml",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = load_yaml(config_path)
        model_cfg = self.config.get("model", {})
        inference_cfg = self.config.get("inference", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = int(model_cfg.get("max_length", 64))
        processor_name = str(
            model_cfg.get("processor_name") or model_cfg.get("name", "OFA-Sys/chinese-clip-vit-large-patch14")
        )
        self.processor = load_processor(processor_name)
        ckpt_path = resolve_path(
            checkpoint_path
            or inference_cfg.get("checkpoint_path", "checkpoints/semantic_model_focal_binary/best.pt")
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model = build_semantic_model_from_config(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.description_provider = build_description_provider(self.config.get("description_provider"))
        self.risk_threshold = float(inference_cfg.get("risk_threshold", 0.65))

    def _build_batch(
        self,
        image_path: str | Path,
        text: str,
        text_description: str,
        meme_description: str,
    ) -> dict[str, torch.Tensor]:
        text_inputs = tokenize_text(self.processor, text, self.max_length)
        text_description_inputs = tokenize_text(self.processor, text_description, self.max_length)
        meme_description_inputs = tokenize_text(self.processor, meme_description, self.max_length)
        pixel_values = process_image(self.processor, image_path)
        batch = {
            "text_input_ids": text_inputs["input_ids"].unsqueeze(0),
            "text_attention_mask": text_inputs["attention_mask"].unsqueeze(0),
            "text_description_input_ids": text_description_inputs["input_ids"].unsqueeze(0),
            "text_description_attention_mask": text_description_inputs["attention_mask"].unsqueeze(0),
            "meme_description_input_ids": meme_description_inputs["input_ids"].unsqueeze(0),
            "meme_description_attention_mask": meme_description_inputs["attention_mask"].unsqueeze(0),
            "pixel_values": pixel_values.unsqueeze(0),
        }
        return {key: value.to(self.device) for key, value in batch.items()}

    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
        text: str = "",
        text_description: str | None = None,
        meme_description: str | None = None,
        use_descriptions: bool | None = None,
    ) -> dict[str, Any]:
        if text_description is None or meme_description is None:
            provided = self.description_provider.describe(image_path, text=text)
            text_description = provided.text_description if text_description is None else text_description
            meme_description = provided.meme_description if meme_description is None else meme_description
        else:
            provided = None

        batch = self._build_batch(
            image_path=image_path,
            text=text,
            text_description=text_description or "",
            meme_description=meme_description or "",
        )
        outputs = self.model(**batch, use_descriptions=use_descriptions)
        type_probabilities = F.softmax(outputs["type_logits"], dim=-1)[0].detach().cpu()
        binary_probabilities = F.softmax(outputs["binary_logits"], dim=-1)[0].detach().cpu()
        modal_probabilities = torch.sigmoid(outputs["modal_logits"])[0].detach().cpu()
        gate = outputs["description_gate"][0].detach().cpu()

        type_id = int(type_probabilities.argmax().item())
        binary_id = int(binary_probabilities.argmax().item())
        harmful_probability = float(binary_probabilities[1].item())
        risk_level = "risk" if harmful_probability >= self.risk_threshold else "normal"
        return {
            "risk_level": risk_level,
            "type_label": TYPE_ID_TO_LABEL[type_id],
            "type_confidence": float(type_probabilities[type_id].item()),
            "binary_label": BINARY_ID_TO_LABEL[binary_id],
            "harmful_probability": harmful_probability,
            "type_probabilities": {
                TYPE_ID_TO_LABEL[idx]: float(type_probabilities[idx].item())
                for idx in range(len(TYPE_ID_TO_LABEL))
            },
            "binary_probabilities": {
                BINARY_ID_TO_LABEL[idx]: float(binary_probabilities[idx].item())
                for idx in range(len(BINARY_ID_TO_LABEL))
            },
            "modal_probabilities": {
                MODAL_LABELS[idx]: float(modal_probabilities[idx].item())
                for idx in range(len(MODAL_LABELS))
            },
            "description_gate": {
                "text_description": float(gate[0].item()),
                "meme_description": float(gate[1].item()),
            },
            "descriptions": {
                "text_description": text_description or "",
                "meme_description": meme_description or "",
                "provider": getattr(provided, "provider", "manual"),
                "error": getattr(provided, "error", None),
            },
        }
