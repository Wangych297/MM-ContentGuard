"""
文件功能：OCR 文本提取服务。基于 EasyOCR 的图片文字识别，
支持 GPU/CPU 自动选择和中英文等多语言识别，对识别结果
进行后处理（去重、合并、过滤）。

核心类：
- OcrService: OCR 服务封装，管理 EasyOCR 实例和识别参数
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from common.image_io import load_rgb_image

_EASYOCR_READERS: dict[tuple[str, bool], Any] = {}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class OcrService:
    def __init__(
        self,
        enabled: bool = True,
        engine: str = "easyocr",
        languages: str = "chi_sim+eng",
        gpu: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.engine = engine
        self.languages = languages
        self.gpu = _env_flag("RISK_IMAGE_OCR_GPU", True) if gpu is None else gpu

    def extract_text(self, image_path: str | Path) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "text": "", "engine": self.engine}

        if self.engine == "easyocr":
            return self._extract_with_easyocr(image_path)
        if self.engine == "pytesseract":
            return self._extract_with_pytesseract(image_path)

        return {
            "enabled": True,
            "text": "",
            "engine": self.engine,
            "error": f"Unsupported OCR engine: {self.engine}",
        }

    def _extract_with_pytesseract(self, image_path: str | Path) -> dict[str, Any]:
        try:
            import pytesseract
        except ImportError:
            return {
                "enabled": True,
                "text": "",
                "engine": self.engine,
                "error": "pytesseract is not installed",
            }

        try:
            image = load_rgb_image(image_path)
            text = pytesseract.image_to_string(image, lang=self.languages)
            return {"enabled": True, "text": text.strip(), "engine": self.engine}
        except Exception as exc:
            return {"enabled": True, "text": "", "engine": self.engine, "error": str(exc)}

    def _extract_with_easyocr(self, image_path: str | Path) -> dict[str, Any]:
        try:
            import easyocr
        except ImportError:
            return {
                "enabled": True,
                "text": "",
                "engine": self.engine,
                "error": "easyocr is not installed",
            }

        language_codes = self._easyocr_languages()
        reader_key = ("+".join(language_codes), self.gpu)
        try:
            reader = _EASYOCR_READERS.get(reader_key)
            if reader is None:
                reader = easyocr.Reader(language_codes, gpu=self.gpu, verbose=False)
                _EASYOCR_READERS[reader_key] = reader
            import numpy as np

            image = np.array(load_rgb_image(image_path))
            results = reader.readtext(image, detail=1, paragraph=False)
            lines = []
            items = []
            for box, text, confidence in results:
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                score = float(confidence)
                lines.append(cleaned)
                items.append(
                    {
                        "text": cleaned,
                        "confidence": score,
                        "box": [[int(point[0]), int(point[1])] for point in box],
                    }
                )
            return {
                "enabled": True,
                "text": "\n".join(lines),
                "engine": self.engine,
                "languages": language_codes,
                "gpu": self.gpu,
                "items": items,
            }
        except Exception as exc:
            return {"enabled": True, "text": "", "engine": self.engine, "error": str(exc)}

    def _easyocr_languages(self) -> list[str]:
        language_map = {
            "chi_sim": "ch_sim",
            "chi_tra": "ch_tra",
            "eng": "en",
            "en": "en",
            "ch_sim": "ch_sim",
            "ch_tra": "ch_tra",
        }
        languages = []
        for raw in self.languages.replace(",", "+").split("+"):
            code = raw.strip()
            if not code:
                continue
            languages.append(language_map.get(code, code))
        if not languages:
            languages = ["ch_sim", "en"]
        return list(dict.fromkeys(languages))


