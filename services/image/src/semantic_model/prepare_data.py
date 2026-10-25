"""
文件功能：语义模型数据准备。将语义 Meme 标注数据按标签分布
进行分层划分，生成训练集、验证集和测试集的 JSON 文件。

核心函数：
- prepare_data: 划分并标准化语义标注数据
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.config import load_yaml, resolve_path
from data.io import write_jsonl
from semantic_model.labels import BINARY_ID_TO_LABEL, TYPE_ID_TO_LABEL


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def _modal_pair(row: dict[str, Any]) -> tuple[int, int]:
    if "text_modal" in row or "image_modal" in row:
        return int(row.get("text_modal", 0)), int(row.get("image_modal", 0))
    modal = row.get("modal", [0, 0])
    if isinstance(modal, list) and len(modal) >= 2:
        return int(modal[0]), int(modal[1])
    return 0, 0


def _normalize_record(row: dict[str, Any], image_dir: Path, skip_missing: bool) -> dict[str, Any] | None:
    file_name = str(row.get("path") or row.get("new_path") or "").strip()
    if not file_name:
        return None
    image_path = (image_dir / file_name).resolve()
    if skip_missing and not image_path.exists():
        return None

    binary_label = int(row.get("label", 0))
    type_label = int(row.get("type", 0))
    text_modal, image_modal = _modal_pair(row)
    return {
        "id": Path(file_name).stem,
        "image_path": str(image_path),
        "file_name": file_name,
        "text": str(row.get("text") or ""),
        "text_description": str(row.get("text_discription") or row.get("text_description") or ""),
        "meme_description": str(row.get("meme_discription") or row.get("meme_description") or ""),
        "binary_label": binary_label,
        "binary_label_name": BINARY_ID_TO_LABEL.get(binary_label, str(binary_label)),
        "type_label": type_label,
        "type_label_name": TYPE_ID_TO_LABEL.get(type_label, str(type_label)),
        "text_modal": text_modal,
        "image_modal": image_modal,
        "target": int(row.get("target", 0) or 0),
        "source": "ToxiCN_MM",
    }


def _stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[int(record["type_label"])].append(record)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)
        val_count = max(1, int(round(len(bucket_records) * val_ratio))) if len(bucket_records) > 1 else 0
        val_records.extend(bucket_records[:val_count])
        train_records.extend(bucket_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "type_labels": dict(sorted(Counter(r["type_label_name"] for r in records).items())),
        "binary_labels": dict(sorted(Counter(r["binary_label_name"] for r in records).items())),
        "modal_pairs": dict(
            sorted(Counter(f'{r["text_modal"]},{r["image_modal"]}' for r in records).items())
        ),
    }


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("project", {}).get("seed", 42))
    data_cfg = config.get("data", {})
    train_json = resolve_path(data_cfg.get("train_json"))
    test_json = resolve_path(data_cfg.get("test_json"))
    image_dir = resolve_path(data_cfg.get("image_dir"))
    output_dir = resolve_path(data_cfg.get("output_dir", "data/processed/semantic_meme"))
    skip_missing = bool(data_cfg.get("skip_missing_images", True))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))

    train_source = [
        record
        for row in _load_json_array(train_json)
        if (record := _normalize_record(row, image_dir, skip_missing)) is not None
    ]
    test_records = [
        record
        for row in _load_json_array(test_json)
        if (record := _normalize_record(row, image_dir, skip_missing)) is not None
    ]
    train_records, val_records = _stratified_split(train_source, val_ratio=val_ratio, seed=seed)

    paths = {
        "train": resolve_path(data_cfg.get("train_file", output_dir / "train.jsonl")),
        "val": resolve_path(data_cfg.get("val_file", output_dir / "val.jsonl")),
        "test": resolve_path(data_cfg.get("test_file", output_dir / "test.jsonl")),
    }
    counts = {
        "train": write_jsonl(paths["train"], train_records),
        "val": write_jsonl(paths["val"], val_records),
        "test": write_jsonl(paths["test"], test_records),
    }
    summary = {
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": counts,
        "splits": {
            "train": _summary(train_records),
            "val": _summary(val_records),
            "test": _summary(test_records),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_meme.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_data(load_yaml(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
