"""
文件功能：分类指标计算。计算图片风险分类的各类别精确率、
召回率、F1 和整体准确率。

核心函数：
- classification_metrics: 计算多分类指标
"""

from __future__ import annotations

from collections import Counter


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    id_to_label: dict[int, str],
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_label": {}}

    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []

    for label_id, label in id_to_label.items():
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label_id and b == label_id)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label_id and b == label_id)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label_id and b != label_id)
        support = sum(1 for a in y_true if a == label_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    label_counts = Counter(y_true)
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "support": total,
        "label_counts": {id_to_label[idx]: count for idx, count in sorted(label_counts.items())},
        "per_label": per_label,
    }

