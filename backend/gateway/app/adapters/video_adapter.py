"""
文件功能：视频审核适配器。封装对视频审核服务的 HTTP 调用，通过
multipart/form-data 上传视频文件并将 PCLMM 多模态模型的预测结果
转换为 ModalityReviewResult。

核心类：
- VideoReviewAdapter: 视频审核适配器，提供 health() 和 review_video_file() 方法
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from app.adapters.base import check_json_health
from app.core.config import GatewaySettings
from app.core.labels import display_label
from app.models.review import ModalityReviewResult, ReviewReason, ServiceHealth


class VideoReviewAdapter:
    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    async def health(self, client: httpx.AsyncClient) -> ServiceHealth:
        return await check_json_health(
            client,
            name="video",
            base_url=self.settings.video_api_url,
            timeout_seconds=10.0,
        )

    async def review_video_file(
        self,
        client: httpx.AsyncClient,
        video_path: str,
        debug: bool = True,
    ) -> ModalityReviewResult:
        url = "%s/predict" % self.settings.video_api_url
        path = Path(video_path)
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, "video/mp4")}
            data = {
                "threshold": str(self.settings.video_threshold),
                "debug": "true" if debug else "false",
            }
            response = await client.post(
                url,
                data=data,
                files=files,
                timeout=self.settings.video_api_timeout,
            )
        response.raise_for_status()
        raw: Dict[str, Any] = response.json()
        return self.from_raw(raw)

    def from_raw(self, raw: Dict[str, Any]) -> ModalityReviewResult:
        is_normal = bool(raw.get("is_normal"))
        score = _optional_float(raw.get("score"))
        category = "normal" if is_normal else "video_abnormal"
        reasons = []
        if not is_normal:
            reasons.append(
                ReviewReason(
                    modality="video",
                    code=category,
                    display=display_label("video", category),
                    score=score,
                    source="video_pclmm",
                )
            )
        return ModalityReviewResult(
            modality="video",
            allow=is_normal,
            risk_level="normal" if is_normal else "risk",
            category=category,
            score=score,
            summary="审核通过" if is_normal else "发送失败：%s" % display_label("video", category),
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
