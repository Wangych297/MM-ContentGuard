"""
文件功能：标签词汇管理。定义风险标签的编码/解码逻辑，管理多任务
标签的索引映射和 IGNORE_INDEX 机制，用于训练时的标签张量构建。

核心类：
- LabelVocab: 标签词汇表，提供标签编解码功能

核心常量：
- IGNORE_INDEX: 忽略标签索引（-100），用于屏蔽无效标签的损失计算
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from risk_text_detection.data.schema import PRIMARY_CATEGORIES

IGNORE_INDEX = -100


DEFAULT_TASKS = ("risk_binary", "primary_category", "toxic_binary", "camouflage_type")

STATIC_LABELS: dict[str, list[str]] = {
    "risk_binary": ["0", "1"],
    "primary_category": list(PRIMARY_CATEGORIES),
    "toxic_binary": ["0", "1"],
    "camouflage_type": [
        "none",
        "unknown",
        "homophone",
        "glyph",
        "number",
        "latin",
        "separator",
        "traditional",
        "mixed",
    ],
}


@dataclass
class TaskLabelVocab:
    task: str
    labels: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    def encode(self, value: Any) -> int:
        if value is None:
            return IGNORE_INDEX
        key = str(value)
        return self.label_to_id.get(key, IGNORE_INDEX)


@dataclass
class LabelVocab:
    tasks: dict[str, TaskLabelVocab]

    @classmethod
    def build(cls, records: Iterable[dict[str, Any]], task_names: Iterable[str]) -> "LabelVocab":
        records = list(records)
        tasks: dict[str, TaskLabelVocab] = {}
        for task in task_names:
            if task in STATIC_LABELS:
                labels = STATIC_LABELS[task]
            else:
                values = {
                    str(record.get("labels", {}).get(task))
                    for record in records
                    if record.get("labels", {}).get(task) is not None
                }
                labels = sorted(values)
            tasks[task] = TaskLabelVocab(task=task, labels=list(labels))
        return cls(tasks=tasks)

    @classmethod
    def from_file(cls, path: Path) -> "LabelVocab":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            tasks={
                task: TaskLabelVocab(task=task, labels=list(spec["labels"]))
                for task, spec in data["tasks"].items()
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": {name: asdict(vocab) for name, vocab in self.tasks.items()}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def num_labels(self) -> dict[str, int]:
        return {task: vocab.num_labels for task, vocab in self.tasks.items()}

    def encode_record(self, record: dict[str, Any]) -> dict[str, int]:
        labels = record.get("labels", {})
        return {task: vocab.encode(labels.get(task)) for task, vocab in self.tasks.items()}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records

