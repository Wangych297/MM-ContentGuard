"""
文件功能：适配器基础工具。提供所有模态适配器共用的 HTTP 健康检查函数。

核心函数：
- check_json_health: 对指定服务发起 GET /health 请求，返回 ServiceHealth 结果
"""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from app.models.review import ServiceHealth


async def check_json_health(
    client: httpx.AsyncClient,
    name: str,
    base_url: str,
    timeout_seconds: float,
) -> ServiceHealth:
    url = "%s/health" % base_url.rstrip("/")
    started_at = time.perf_counter()
    try:
        response = await client.get(url, timeout=timeout_seconds)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        raw: Dict[str, Any]
        try:
            raw = response.json()
        except ValueError:
            raw = {"text": response.text[:500]}
        return ServiceHealth(
            ok=response.is_success,
            name=name,
            url=url,
            status_code=response.status_code,
            latency_ms=latency_ms,
            raw=raw,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return ServiceHealth(
            ok=False,
            name=name,
            url=url,
            latency_ms=latency_ms,
            error=str(exc),
        )
