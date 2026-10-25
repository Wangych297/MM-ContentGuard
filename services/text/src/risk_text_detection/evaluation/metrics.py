"""
文件功能：评测指标计算。计算文本风险检测模型的分类指标，包括
精确率（Precision）、召回率（Recall）、F1 和准确率（Accuracy）。

核心函数：
- compute_metrics: 计算各标签的精确率、召回率、F1 和准确率
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from risk_text_detection.training.labeling import IGNORE_INDEX


def update_predictions(
    store: dict[str, dict[str, list[int]]],
    logits,
    labels,
) -> None:
    for task, task_logits in logits.items():
        if task not in labels:
            continue
        preds = task_logits.detach().argmax(dim=-1).cpu().tolist()
        gold = labels[task].detach().cpu().tolist()
        for pred, target in zip(preds, gold):
            if target == IGNORE_INDEX:
                continue
            store[task]["pred"].append(int(pred))
            store[task]["gold"].append(int(target))


def empty_prediction_store() -> dict[str, dict[str, list[int]]]:
    return defaultdict(lambda: {"pred": [], "gold": []})


def compute_metrics(store: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for task, values in store.items():
        gold = values["gold"]
        pred = values["pred"]
        if not gold:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold,
            pred,
            average="macro",
            zero_division=0,
        )
        weighted = precision_recall_fscore_support(
            gold,
            pred,
            average="weighted",
            zero_division=0,
        )[2]
        results[task] = {
            "accuracy": float(accuracy_score(gold, pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "weighted_f1": float(weighted),
            "support": float(len(gold)),
            "num_classes_seen": float(len(np.unique(gold))),
        }
    return results

