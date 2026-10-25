"""
文件功能：图片审核适配器。封装对图片审核服务的 HTTP 调用，将图片路径
和 OCR 开关发送至图片审核服务，并将返回的检测结果转换为 ModalityReviewResult。

核心类：
- ImageReviewAdapter: 图片审核适配器，提供 health() 和 review_image_path() 方法

核心函数：
- _build_reasons: 解析图片审核原始结果，构建审核原因列表
- _extract_image_score: 从原始响应中提取模型置信度分数
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.adapters.base import check_json_health
from app.core.config import GatewaySettings
from app.core.labels import display_label
from app.models.review import ModalityReviewResult, ReviewReason, ServiceHealth


class ImageReviewAdapter:
    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    async def health(self, client: httpx.AsyncClient) -> ServiceHealth:
        return await check_json_health(
            client,
            name="image",
            base_url=self.settings.image_api_url,
            timeout_seconds=self.settings.image_api_timeout,
        )

    async def review_image_path(
        self,
        client: httpx.AsyncClient,
        image_path: str,
        use_ocr: Optional[bool] = None,
    ) -> ModalityReviewResult:
        url = "%s/image/check" % self.settings.image_api_url
        response = await client.post(
            url,
            json={
                "image_path": image_path,
                "use_ocr": self.settings.image_use_ocr if use_ocr is None else use_ocr,
            },
            timeout=self.settings.image_api_timeout,
        )
        response.raise_for_status()
        raw: Dict[str, Any] = response.json()
        return self.from_raw(raw)

    def from_raw(self, raw: Dict[str, Any]) -> ModalityReviewResult:
        risk_level = str(raw.get("risk_level") or "normal")
        category = str(raw.get("risk_category") or "safe")
        allow = risk_level == "normal"
        raw_reasons = raw.get("reasons") or []
        reasons = _build_reasons(category, raw_reasons, allow)

        return ModalityReviewResult(
            modality="image",
            allow=allow,
            risk_level=risk_level,
            category="normal" if allow else category,
            score=_extract_image_score(raw),
            summary="审核通过" if allow else "发送失败：%s" % display_label("image", category),
            reasons=reasons,
            raw=raw,
        )


def _build_reasons(category: str, raw_reasons: List[Any], allow: bool) -> List[ReviewReason]:
    if allow:
        return []

    if not raw_reasons:
        return [
            ReviewReason(
                modality="image",
                code=category,
                display=display_label("image", category),
                source="image_pipeline",
            )
        ]

    reasons = []
    for raw_reason in raw_reasons:
        raw_text = str(raw_reason)
        code = category
        source = "image_pipeline"
        if ":" in raw_text:
            source, code = raw_text.split(":", 1)
            if "=" in code:
                code = code.split("=", 1)[0]
        reasons.append(
            ReviewReason(
                modality="image",
                code=code or category,
                display=display_label("image", code or category),
                source=source,
                raw_reason=raw_text,
            )
        )
    return reasons


def _extract_image_score(raw: Dict[str, Any]) -> Optional[float]:
    detectors = raw.get("detectors") or {}
    image_model = detectors.get("image_model") or {}
    for key in ("model_confidence", "model_risk_probability"):
        if key in image_model:
            try:
                return float(image_model[key])
            except (TypeError, ValueError):
                return None
    return None
