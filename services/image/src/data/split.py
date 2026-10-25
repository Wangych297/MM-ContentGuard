"""
文件功能：数据集划分。按标签分布对图片清单进行分层随机划分，
生成训练集、验证集和测试集的 JSONL 文件。

核心函数：
- split_manifest: 对清单记录执行分层划分
- main: 命令行入口
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from common.config import load_yaml, resolve_path
from data.io import read_jsonl, write_jsonl


def _stratified_split(
    records: list[dict],
    label_key: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[str(record[label_key])].append(record)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for label, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 3 else 0
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0
        if n_test + n_val >= n_total:
            n_test = max(0, n_total // 5)
            n_val = max(0, n_total // 5)

        for record in shuffled[:n_test]:
            item = dict(record)
            item["split"] = "test"
            test.append(item)
        for record in shuffled[n_test : n_test + n_val]:
            item = dict(record)
            item["split"] = "val"
            val.append(item)
        for record in shuffled[n_test + n_val :]:
            item = dict(record)
            item["split"] = "train"
            train.append(item)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _describe(name: str, records: list[dict]) -> None:
    by_label = Counter(record["label"] for record in records)
    print(f"{name}: {len(records)} records, labels={dict(sorted(by_label.items()))}")


def split_manifest(config: dict) -> tuple[Path, Path, Path]:
    project_seed = int(config.get("project", {}).get("seed", 42))
    data_cfg = config.get("data", {})
    manifest_path = resolve_path(data_cfg.get("manifest_path", "data/processed/image_manifest.jsonl"))
    split_dir = resolve_path(data_cfg.get("split_dir", "data/processed/splits"))
    split_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(manifest_path))
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    train, val, test = _stratified_split(
        records,
        label_key="label",
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        seed=project_seed,
    )

    train_path = resolve_path(data_cfg.get("train_file", split_dir / "image_train.jsonl"))
    val_path = resolve_path(data_cfg.get("val_file", split_dir / "image_val.jsonl"))
    test_path = resolve_path(data_cfg.get("test_file", split_dir / "image_test.jsonl"))

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    _describe("train", train)
    _describe("val", val)
    _describe("test", test)
    return train_path, val_path, test_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_classifier_dinov3_large.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    split_manifest(config)


if __name__ == "__main__":
    main()



