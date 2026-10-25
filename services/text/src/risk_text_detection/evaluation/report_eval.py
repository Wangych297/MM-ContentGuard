"""
文件功能：评测报告生成。加载模型预测结果，按标签维度生成详细的
分类评测报告，包括各类别的精确率、召回率、F1 和混淆统计。

核心函数：
- report_evaluation: 生成详细的文本风险检测评测报告
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from risk_text_detection.inference.predictor import DEFAULT_CHECKPOINT_DIR, RiskTextPredictor
from risk_text_detection.training.labeling import LabelVocab, read_jsonl


DEFAULT_TASKS = ("risk_binary", "primary_category", "toxic_binary", "camouflage_type")


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _gold_label(record: dict[str, Any], task: str) -> str | None:
    value = record.get("labels", {}).get(task)
    if value is None:
        return None
    return str(value)


def _predict_records(
    predictor: RiskTextPredictor,
    records: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for batch in _chunks(records, batch_size):
        predictions.extend(predictor.predict_batch([record["text"] for record in batch]))
    return predictions


def _task_pairs(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    task: str,
) -> tuple[list[str], list[str]]:
    gold: list[str] = []
    pred: list[str] = []
    for record, prediction in zip(records, predictions):
        target = _gold_label(record, task)
        if target is None:
            continue
        task_pred = prediction.get("raw_argmax", {}).get(task)
        if task_pred is None:
            continue
        gold.append(target)
        pred.append(str(task_pred))
    return gold, pred


def _metric_dict(gold: list[str], pred: list[str]) -> dict[str, float | int]:
    if not gold:
        return {
            "support": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        gold,
        pred,
        average="weighted",
        zero_division=0,
    )[2]
    return {
        "support": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
    }


def _write_classification_report(
    output_dir: Path,
    task: str,
    labels: list[str],
    gold: list[str],
    pred: list[str],
) -> None:
    if not gold:
        pd.DataFrame(
            columns=["label", "precision", "recall", "f1-score", "support"]
        ).to_csv(output_dir / f"{task}_classification_report.csv", index=False)
        return
    report = classification_report(
        gold,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
        else:
            rows.append({"label": label, "value": values})
    pd.DataFrame(rows).to_csv(output_dir / f"{task}_classification_report.csv", index=False)


def _write_confusion_matrix(
    output_dir: Path,
    task: str,
    labels: list[str],
    gold: list[str],
    pred: list[str],
) -> None:
    matrix = confusion_matrix(gold, pred, labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "gold\\pred"
    frame.to_csv(output_dir / f"{task}_confusion_matrix.csv")


def _evaluate_subset(
    name: str,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    tasks: Iterable[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "samples": len(records),
        "source_counts": dict(Counter(r.get("metadata", {}).get("source") for r in records)),
        "tasks": {},
    }
    for task in tasks:
        gold, pred = _task_pairs(records, predictions, task)
        result["tasks"][task] = _metric_dict(gold, pred)
    return result


def _source_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    tasks: Iterable[str],
) -> pd.DataFrame:
    rows = []
    sources = sorted({record.get("metadata", {}).get("source", "unknown") for record in records})
    for source in sources:
        idxs = [
            idx
            for idx, record in enumerate(records)
            if record.get("metadata", {}).get("source", "unknown") == source
        ]
        source_records = [records[idx] for idx in idxs]
        source_predictions = [predictions[idx] for idx in idxs]
        for task in tasks:
            gold, pred = _task_pairs(source_records, source_predictions, task)
            metrics = _metric_dict(gold, pred)
            rows.append({"source": source, "task": task, **metrics})
    return pd.DataFrame(rows)


def _threshold_scan(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    thresholds: list[float],
) -> pd.DataFrame:
    gold = []
    risk_probs = []
    for record, prediction in zip(records, predictions):
        target = _gold_label(record, "risk_binary")
        if target is None:
            continue
        gold.append(1 if target == "1" else 0)
        risk_probs.append(float(prediction.get("risk_probability", 0.0)))

    rows = []
    for threshold in thresholds:
        pred = [1 if prob >= threshold else 0 for prob in risk_probs]
        if gold:
            precision, recall, f1, _ = precision_recall_fscore_support(
                gold,
                pred,
                labels=[1],
                average="binary",
                zero_division=0,
            )
            accuracy = float(accuracy_score(gold, pred))
        else:
            precision = recall = f1 = accuracy = 0.0
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "risk_precision": float(precision),
                "risk_recall": float(recall),
                "risk_f1": float(f1),
                "pred_risk": int(sum(pred)),
                "gold_risk": int(sum(gold)),
                "support": len(gold),
            }
        )
    return pd.DataFrame(rows)


def _subset_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ast_records = [r for r in records if r.get("metadata", {}).get("source") == "ast"]
    ast_risk = [r for r in ast_records if r.get("labels", {}).get("risk_binary") == 1]
    non_ast_safe = [
        r
        for r in records
        if r.get("metadata", {}).get("source") != "ast"
        and r.get("labels", {}).get("risk_binary") == 0
    ]
    n = min(len(ast_risk), len(non_ast_safe))
    return {
        "ast_only": ast_records,
        "ast_risk_plus_non_ast_safe": ast_risk[:n] + non_ast_safe[:n],
        "spam_message_only": [
            r for r in records if r.get("metadata", {}).get("source") == "spam_message"
        ],
        "fbs_sms_only": [r for r in records if r.get("metadata", {}).get("source") == "fbs"],
    }


def run_report_eval(
    checkpoint_dir: str | Path,
    test_file: str | Path,
    output_dir: str | Path,
    threshold_file: str | Path | None = None,
    batch_size: int = 16,
    limit_records: int | None = None,
    limit_threshold_records: int | None = None,
    tasks: Iterable[str] = DEFAULT_TASKS,
    thresholds: list[float] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or [round(value / 100, 2) for value in range(30, 91, 5)]
    task_list = list(tasks)

    label_vocab = LabelVocab.from_file(checkpoint_dir / "label_vocab.json")
    predictor = RiskTextPredictor(checkpoint_dir=checkpoint_dir, device=device)

    test_records = read_jsonl(Path(test_file), limit=limit_records)
    test_predictions = _predict_records(predictor, test_records, batch_size=batch_size)

    overall = _evaluate_subset("overall", test_records, test_predictions, task_list)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for task in task_list:
        gold, pred = _task_pairs(test_records, test_predictions, task)
        labels = label_vocab.tasks[task].labels
        _write_classification_report(output_dir, task, labels, gold, pred)
        _write_confusion_matrix(output_dir, task, labels, gold, pred)

    source_frame = _source_metrics(test_records, test_predictions, task_list)
    source_frame.to_csv(output_dir / "source_metrics.csv", index=False)

    subset_results = []
    for name, subset in _subset_records(test_records).items():
        if not subset:
            continue
        subset_predictions = _predict_records(predictor, subset, batch_size=batch_size)
        subset_results.append(_evaluate_subset(name, subset, subset_predictions, task_list))
    (output_dir / "subset_metrics.json").write_text(
        json.dumps(subset_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    threshold_records = (
        read_jsonl(Path(threshold_file), limit=limit_threshold_records or limit_records)
        if threshold_file
        else test_records
    )
    threshold_predictions = (
        _predict_records(predictor, threshold_records, batch_size=batch_size)
        if threshold_file
        else test_predictions
    )
    threshold_frame = _threshold_scan(threshold_records, threshold_predictions, thresholds)
    threshold_frame.to_csv(output_dir / "risk_threshold_scan.csv", index=False)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "test_file": str(test_file),
        "threshold_file": str(threshold_file) if threshold_file else str(test_file),
        "output_dir": str(output_dir),
        "limit_records": limit_records,
        "limit_threshold_records": limit_threshold_records or limit_records,
        "overall": overall,
        "artifacts": [
            "overall_metrics.json",
            "source_metrics.csv",
            "subset_metrics.json",
            "risk_threshold_scan.csv",
            *[f"{task}_classification_report.csv" for task in task_list],
            *[f"{task}_confusion_matrix.csv" for task in task_list],
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--test-file", default="data/processed/lab_full/test.jsonl")
    parser.add_argument("--threshold-file", default="data/processed/lab_full/dev.jsonl")
    parser.add_argument("--output-dir", default="reports/lab_full_eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--limit-threshold-records", type=int)
    parser.add_argument("--device")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[round(value / 100, 2) for value in range(30, 91, 5)],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_report_eval(
        checkpoint_dir=args.checkpoint_dir,
        test_file=args.test_file,
        threshold_file=args.threshold_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        limit_records=args.limit_records,
        limit_threshold_records=args.limit_threshold_records,
        tasks=args.tasks,
        thresholds=args.thresholds,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
