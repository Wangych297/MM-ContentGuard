"""
文件功能：图片数据模式定义。定义图片审核流程中的数据模型，
包括图片记录、规则匹配结果和检测结果的数据类。

核心类：
- ImageRecord: 图片样本记录，包含路径和标签信息
- RuleHit: 规则命中结果，记录命中的规则类型和匹配文本
- DetectionResult: 单个检测器的检测结果
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImageRecord:
    id: str
    image_path: str
    label: str
    coarse_label: str
    is_safe: bool
    source: str
    original_label: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleHit:
    name: str
    value: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    detector: str
    label: str
    risk_level: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


