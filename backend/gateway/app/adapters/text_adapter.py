"""
文件功能：文本审核适配器。封装对文本审核服务的 HTTP 调用，支持
risk_text（BERT 模型）和 tfidf_svm（传统机器学习）两种后端，
将下游响应统一转换为 ModalityReviewResult。

核心类：
- TextReviewAdapter: 文本审核适配器，提供 health() 和 review_text() 方法
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.adapters.base import check_json_health
from app.core.config import GatewaySettings
from app.core.labels import display_label
from app.models.review import ModalityReviewResult, ReviewReason, ServiceHealth


class TextReviewAdapter:
    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    async def health(self, client: httpx.AsyncClient) -> ServiceHealth:
        return await check_json_health(
            client,
            name="text",
            base_url=self.settings.text_api_url,
            timeout_seconds=self.settings.text_api_timeout,
        )

    async def review_text(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> ModalityReviewResult:
        if self.settings.text_api_backend == "tfidf_svm":
            return await self._review_text_with_tfidf_svm(client, text)
        if self.settings.text_api_backend != "risk_text":
            raise ValueError("Unsupported TEXT_API_BACKEND: %s" % self.settings.text_api_backend)

        return await self._review_text_with_risk_text(client, text)

    async def _review_text_with_risk_text(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> ModalityReviewResult:
        url = "%s/text/check" % self.settings.text_api_url
        response = await client.post(
            url,
            json={
                "text": text,
                "risk_threshold": self.settings.text_risk_threshold,
                "suspicious_threshold": self.settings.text_suspicious_threshold,
            },
            timeout=self.settings.text_api_timeout,
        )
        response.raise_for_status()
        raw: Dict[str, Any] = response.json()
        return self.from_risk_text_raw(raw)

    async def _review_text_with_tfidf_svm(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> ModalityReviewResult:
        url = "%s/predict" % self.settings.text_api_url
        response = await client.post(
            url,
            json={"text": text},
            timeout=self.settings.text_api_timeout,
        )
        response.raise_for_status()
        is_risk = bool(response.json())
        raw = {
            "backend": "tfidf_svm",
            "text": text,
            "is_violation": is_risk,
            "risk_level": "risk" if is_risk else "normal",
            "primary_category": "spam_general" if is_risk else "normal",
        }
        return self.from_tfidf_svm_raw(raw)

    def from_risk_text_raw(self, raw: Dict[str, Any]) -> ModalityReviewResult:
        risk_level = str(raw.get("risk_level") or "normal")
        category = str(raw.get("primary_category") or "unknown")
        score = _optional_float(raw.get("risk_probability"))
        allow = risk_level == "normal"

        reasons = []
        if not allow:
            reasons.append(
                ReviewReason(
                    modality="text",
                    code=category,
                    display=display_label("text", category),
                    score=score,
                    source="text_model",
                )
            )

        return ModalityReviewResult(
            modality="text",
            allow=allow,
            risk_level=risk_level,
            category="normal" if allow else category,
            score=score,
            summary="审核通过" if allow else "发送失败：%s" % display_label("text", category),
            reasons=reasons,
            raw=raw,
        )

    def from_tfidf_svm_raw(self, raw: Dict[str, Any]) -> ModalityReviewResult:
        risk_level = str(raw.get("risk_level") or "normal")
        category = str(raw.get("primary_category") or "unknown")
        allow = risk_level == "normal"

        reasons = []
        if not allow:
            reasons.append(
                ReviewReason(
                    modality="text",
                    code=category,
                    display=display_label("text", category),
                    source="tfidf_svm_text_model",
                )
            )

        return ModalityReviewResult(
            modality="text",
            allow=allow,
            risk_level=risk_level,
            category="normal" if allow else category,
            summary="审核通过" if allow else "发送失败：%s" % display_label("text", category),
            reasons=reasons,
            raw=raw,
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
