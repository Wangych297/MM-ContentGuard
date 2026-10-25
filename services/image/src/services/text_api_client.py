"""
文件功能：文本服务 HTTP 客户端。将 OCR 提取的图片文本转发至
文本审核服务进行风险检测，作为图片流水线中的文本审核环节。

核心类：
- TextRiskClient: 文本风险服务 HTTP 客户端
"""

from __future__ import annotations

from typing import Any

import requests


class TextRiskClient:
    def __init__(self, enabled: bool, url: str, timeout_seconds: float = 8.0) -> None:
        self.enabled = enabled
        self.url = url
        self.timeout_seconds = timeout_seconds

    def check_text(
        self,
        text: str,
        risk_threshold: float | None = None,
        suspicious_threshold: float | None = None,
    ) -> dict[str, Any]:
        text = text.strip()
        if not self.enabled:
            return {"enabled": False, "text": text}
        if not text:
            return {"enabled": True, "text": text, "skipped": "empty_text"}

        payload: dict[str, Any] = {"text": text}
        if risk_threshold is not None:
            payload["risk_threshold"] = risk_threshold
        if suspicious_threshold is not None:
            payload["suspicious_threshold"] = suspicious_threshold

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            result = response.json()
            result["enabled"] = True
            return result
        except Exception as exc:
            return {
                "enabled": True,
                "text": text,
                "error": str(exc),
                "risk_binary": "unknown",
                "risk_level": "unknown",
            }

