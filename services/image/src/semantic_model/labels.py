"""
文件功能：语义模型标签定义。定义梗图/表情包语义分类的标签体系，
包括语义类型标签、二分类标签和模态标签。

核心常量：
- SEMANTIC_TYPE_LABELS: 语义类型标签（安全/针对性冒犯/色情/一般冒犯/消极）
- BINARY_LABELS: 二分类标签
- MODAL_LABELS: 模态标签
"""

from __future__ import annotations

SEMANTIC_TYPE_LABELS = [
    "safe",
    "targeted_offense",
    "sexual",
    "general_offense",
    "dispirited",
]

BINARY_LABELS = ["safe", "harmful"]
MODAL_LABELS = ["text_risk", "image_risk"]

TYPE_LABEL_TO_ID = {label: idx for idx, label in enumerate(SEMANTIC_TYPE_LABELS)}
TYPE_ID_TO_LABEL = {idx: label for idx, label in enumerate(SEMANTIC_TYPE_LABELS)}
BINARY_ID_TO_LABEL = {idx: label for idx, label in enumerate(BINARY_LABELS)}
