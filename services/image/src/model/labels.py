"""
文件功能：图片标签定义。定义图片风险分类的标签体系，包括
标签列表、标签与索引的双向映射和风险标签判断逻辑。

核心常量：
- IMAGE_LABELS: 图片风险类别标签列表
- RISK_LABELS: 风险标签集合

核心函数：
- label_to_id / id_to_label: 标签名与索引的转换
- is_risk_label: 判断标签是否为风险类别
"""

from __future__ import annotations

IMAGE_LABELS = [
    "safe",
    "sexual",
    "violence",
    "weapon",
    "gambling",
    "drug",
    "regulated_goods",
]

RISK_LABELS = {label for label in IMAGE_LABELS if label != "safe"}

LABEL_TO_ID = {label: idx for idx, label in enumerate(IMAGE_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def label_to_id(label: str) -> int:
    if label not in LABEL_TO_ID:
        raise KeyError(f"Unknown image label: {label}")
    return LABEL_TO_ID[label]


def id_to_label(label_id: int) -> str:
    if label_id not in ID_TO_LABEL:
        raise KeyError(f"Unknown image label id: {label_id}")
    return ID_TO_LABEL[label_id]


def is_risk_label(label: str) -> bool:
    return label in RISK_LABELS


