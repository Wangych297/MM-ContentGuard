"""
文件功能：网关健康检查端点。并发查询文本、图片、视频三个下游服务的
/health 接口，汇总返回各服务的可用状态和延迟信息。

核心接口：
- GET /api/health: 返回网关及各下游服务的健康状态
"""

from __future__ import annotations

import asyncio
from typing import Dict

import httpx
from fastapi import APIRouter

from app.adapters.image_adapter import ImageReviewAdapter
from app.adapters.text_adapter import TextReviewAdapter
from app.adapters.video_adapter import VideoReviewAdapter
from app.core.config import get_settings
from app.models.review import GatewayHealth, ServiceHealth


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=GatewayHealth)
async def health() -> GatewayHealth:
    settings = get_settings()
    async with httpx.AsyncClient(trust_env=False) as client:
        text_adapter = TextReviewAdapter(settings)
        image_adapter = ImageReviewAdapter(settings)
        video_adapter = VideoReviewAdapter(settings)
        text, image, video = await asyncio.gather(
            text_adapter.health(client),
            image_adapter.health(client),
            video_adapter.health(client),
        )

    services: Dict[str, ServiceHealth] = {
        "text": text,
        "image": image,
        "video": video,
    }
    return GatewayHealth(
        ok=all(service.ok for service in services.values()),
        services=services,
    )
