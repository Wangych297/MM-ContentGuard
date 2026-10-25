"""
文件功能：数据模式定义。定义文本风险检测的数据结构，包括风险标签
枚举、RiskRecord 样本记录和文本清洗/哈希工具函数。

核心类：
- RiskLabels: 多任务风险标签数据类
- RiskRecord: 单条文本风险样本记录

核心常量：
- PRIMARY_CATEGORIES: 风险主类别枚举
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


PRIMARY_CATEGORIES = (
    "safe",
    "spam_general",
    "ad_promotion",
    "loan_ad",
    "fraud_phishing",
    "gambling",
    "porn_sexual",
    "illegal_service",
    "political_propaganda",
    "toxicity_abuse",
    "hate_bias",
    "sex_harm",
    "review_spam",
    "other_risk",
)


@dataclass
class RiskLabels:
    risk_binary: int | None
    primary_category: str | None
    risk_tags: list[str] = field(default_factory=list)
    toxic_binary: int | None = None
    toxic_type: str | None = None
    camouflage_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskRecord:
    id: str
    text: str
    labels: RiskLabels
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = self.labels.to_dict()
        return data


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str, max_chars: int = 512) -> str:
    normalized = " ".join(str(text).strip().split())
    if max_chars and len(normalized) > max_chars:
        return normalized[:max_chars]
    return normalized

