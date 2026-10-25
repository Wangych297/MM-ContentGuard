"""
文件功能：图片审核流水线。编排完整的图片审核流程：OCR 文本提取 →
规则检测器（联系方式、二维码、敏感词、URL）→ 视觉模型预测 →
语义模型预测 → 文本风险转发 → 综合决策，返回结构化审核结果。

核心类：
- ImageRiskPipeline: 图片风险审核流水线，串联所有检测和模型环节
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from common.config import load_yaml
from pipeline.decision import decide_image_risk
from detectors.contact import detect_contacts
from detectors.qrcode import detect_qrcodes
from detectors.sensitive_words import detect_sensitive_words
from detectors.url import detect_urls
from services.ocr import OcrService
from services.text_api_client import TextRiskClient
from services.translator import TextTranslator
from model.predictor import ImageModelPredictor


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


class ImageRiskPipeline:
    def __init__(
        self,
        config_path: str | Path = "configs/image_classifier_dinov3_large.yaml",
        load_visual: bool = True,
    ) -> None:
        self.config_path = config_path
        self.config = load_yaml(config_path)
        self.image_model = ImageModelPredictor(config_path=config_path) if load_visual else None

        ocr_cfg = self.config.get("ocr", {})
        self.ocr = OcrService(
            enabled=bool(ocr_cfg.get("enabled", True)),
            engine=str(ocr_cfg.get("engine", "pytesseract")),
            languages=str(ocr_cfg.get("languages", "chi_sim+eng")),
            gpu=_env_flag("RISK_IMAGE_OCR_GPU", True),
        )
        self.translator = TextTranslator(enabled=False)

        text_cfg = self.config.get("text_api", {})
        text_api_url = os.getenv("RISK_IMAGE_TEXT_API_URL") or str(
            text_cfg.get("url", "http://127.0.0.1:8000/text/check")
        )
        self.text_client = TextRiskClient(
            enabled=bool(text_cfg.get("enabled", True)),
            url=text_api_url,
            timeout_seconds=_env_float(
                "RISK_IMAGE_TEXT_API_TIMEOUT",
                float(text_cfg.get("timeout_seconds", 8)),
            ),
        )
        semantic_cfg = self.config.get("semantic_model", {})
        self.semantic_enabled = _env_flag(
            "RISK_IMAGE_SEMANTIC_ENABLED",
            bool(semantic_cfg.get("enabled", False)),
        )
        self.semantic_run_when_stage1_safe = _env_flag(
            "RISK_IMAGE_SEMANTIC_RUN_WHEN_STAGE1_SAFE",
            bool(semantic_cfg.get("run_when_stage1_safe", True)),
        )
        self.semantic_use_descriptions = bool(semantic_cfg.get("use_descriptions", True))
        self.semantic_model = None
        if self.semantic_enabled:
            from semantic_model.predictor import SemanticMemePredictor

            self.semantic_model = SemanticMemePredictor(
                config_path=semantic_cfg.get("config", "configs/semantic_meme.yaml"),
                checkpoint_path=semantic_cfg.get("checkpoint_path"),
            )

    def predict(self, image_path: str | Path, use_ocr: bool = True) -> dict[str, Any]:
        total_started_at = time.perf_counter()
        timings: dict[str, float | bool | int] = {}
        image_path = str(Path(image_path).resolve())

        started_at = time.perf_counter()
        model_result = self.image_model.predict(image_path) if self.image_model is not None else {}
        timings["image_model"] = round(time.perf_counter() - started_at, 4)

        started_at = time.perf_counter()
        ocr_result = self.ocr.extract_text(image_path) if use_ocr else {"enabled": False, "text": ""}
        timings["ocr"] = round(time.perf_counter() - started_at, 4)
        timings["ocr_enabled"] = bool(use_ocr)
        ocr_text = str(ocr_result.get("text") or "")
        translated = self.translator.translate_to_chinese(ocr_text)
        text_for_check = str(translated.get("translated_text") or ocr_text)

        inference_cfg = self.config.get("inference", {})
        started_at = time.perf_counter()
        text_result = self.text_client.check_text(
            text_for_check,
            risk_threshold=float(inference_cfg.get("text_risk_threshold", 0.7)),
            suspicious_threshold=float(inference_cfg.get("text_suspicious_threshold", 0.4)),
        )
        timings["text_risk"] = round(time.perf_counter() - started_at, 4)
        timings["ocr_text_chars"] = len(text_for_check)

        started_at = time.perf_counter()
        rule_cfg = self.config.get("rules", {})
        rule_hits = []
        if rule_cfg.get("enabled", True):
            rule_hits.extend(hit.to_dict() for hit in detect_urls(ocr_text))
            rule_hits.extend(hit.to_dict() for hit in detect_contacts(ocr_text))
            rule_hits.extend(
                hit.to_dict()
                for hit in detect_sensitive_words(ocr_text, rule_cfg.get("sensitive_words"))
            )
            if rule_cfg.get("qrcode_enabled", True):
                rule_hits.extend(hit.to_dict() for hit in detect_qrcodes(image_path))
        timings["rules"] = round(time.perf_counter() - started_at, 4)

        started_at = time.perf_counter()
        preliminary_decision = decide_image_risk(model_result, text_result, rule_hits)
        timings["preliminary_decision"] = round(time.perf_counter() - started_at, 4)
        semantic_result: dict[str, Any] = {"enabled": False}
        should_run_semantic = (
            self.semantic_model is not None
            and self.semantic_run_when_stage1_safe
            and preliminary_decision.get("route") == "send_to_semantic_model"
        )
        if should_run_semantic:
            semantic_result = {
                "enabled": True,
                **self.semantic_model.predict(
                    image_path,
                    text=text_for_check,
                    use_descriptions=self.semantic_use_descriptions,
                ),
            }
        timings["semantic_model"] = round(time.perf_counter() - started_at, 4)
        timings["semantic_model_run"] = bool(should_run_semantic)

        started_at = time.perf_counter()
        decision = decide_image_risk(model_result, text_result, rule_hits, semantic_result)
        timings["final_decision"] = round(time.perf_counter() - started_at, 4)
        timings["total"] = round(time.perf_counter() - total_started_at, 4)
        return {
            **decision,
            "image_path": image_path,
            "timings": timings,
            "detectors": {
                "image_model": model_result,
                "ocr": ocr_result,
                "translation": translated,
                "text_risk": text_result,
                "rules": rule_hits,
                "semantic_model": semantic_result,
            },
        }




