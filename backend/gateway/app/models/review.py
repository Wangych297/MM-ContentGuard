"""
文件功能：审核结果数据模型。定义服务健康状态、审核原因和模态审核结果
的 Pydantic 数据模型，用于网关与下游服务之间的数据交换。

核心类：
- ServiceHealth: 单个下游服务的健康状态模型
- GatewayHealth: 网关聚合健康状态模型
- ReviewReason: 单条审核不通过原因的模型
- ModalityReviewResult: 单个模态的审核结果模型
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    ok: bool
    name: str
    url: str
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class GatewayHealth(BaseModel):
    ok: bool
    services: Dict[str, ServiceHealth]


class ReviewReason(BaseModel):
    modality: str
    code: str
    display: str
    score: Optional[float] = None
    source: Optional[str] = None
    raw_reason: Optional[str] = None


class ModalityReviewResult(BaseModel):
    modality: str
    allow: bool
    risk_level: str
    category: str = "normal"
    score: Optional[float] = None
    summary: str = "审核通过"
    reasons: List[ReviewReason] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)
