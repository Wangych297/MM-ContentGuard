"""
文件功能：语义模型指标计算。计算多标签分类和语义分类的评测指标，
支持宏平均和微平均的精确率、召回率、F1。

核心函数：
- multilabel_metrics: 多标签分类指标计算
- semantic_metrics: 语义分类综合指标计算
"""

from __future__ import annotations

from model.metrics import classification_metrics
from semantic_model.labels import BINARY_ID_TO_LABEL, MODAL_LABELS, TYPE_ID_TO_LABEL


def multilabel_metrics(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    labels: list[str],
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {"macro_f1": 0.0, "exact_match": 0.0, "per_label": {}}

    exact_match = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []

    for idx, label in enumerate(labels):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a[idx] == 1 and b[idx] == 1)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a[idx] == 0 and b[idx] == 1)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a[idx] == 1 and b[idx] == 0)
        support = sum(1 for a in y_true if a[idx] == 1)
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

    return {
        "macro_f1": sum(f1_values) / len(f1_values),
        "exact_match": exact_match,
        "per_label": per_label,
    }


def semantic_metrics(
    type_true: list[int],
    type_pred: list[int],
    binary_true: list[int],
    binary_pred: list[int],
    modal_true: list[list[int]],
    modal_pred: list[list[int]],
) -> dict:
    type_scores = classification_metrics(type_true, type_pred, TYPE_ID_TO_LABEL)
    binary_scores = classification_metrics(binary_true, binary_pred, BINARY_ID_TO_LABEL)
    modal_scores = multilabel_metrics(modal_true, modal_pred, MODAL_LABELS)
    return {
        "type_accuracy": type_scores["accuracy"],
        "type_macro_f1": type_scores["macro_f1"],
        "type_label_counts": type_scores.get("label_counts", {}),
        "type_per_label": type_scores["per_label"],
        "binary_accuracy": binary_scores["accuracy"],
        "binary_macro_f1": binary_scores["macro_f1"],
        "binary_label_counts": binary_scores.get("label_counts", {}),
        "binary_per_label": binary_scores["per_label"],
        "modal_macro_f1": modal_scores["macro_f1"],
        "modal_exact_match": modal_scores["exact_match"],
        "modal_per_label": modal_scores["per_label"],
    }
