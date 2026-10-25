"""
文件功能：构建统一数据集。将多个原始数据源（AST 标注数据、垃圾短信
数据等）合并转换为统一的训练清单文件，支持标签映射和数据采样。

核心函数：
- build_unified_dataset: 合并多数据源构建统一训练数据集
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .converters import (
    load_ast,
    load_chinese_offensive,
    load_cold,
    load_fbs,
    load_spam_message,
    load_toxicn,
    write_jsonl,
)
from .schema import RiskRecord


LOADERS = {
    "spam_message": lambda root, max_chars: load_spam_message(root / "data/raw/spam_message/spam_message_labeled.txt", max_chars),
    "ast": lambda root, max_chars: load_ast(root / "data/raw/ast/dataset.txt", max_chars),
    "fbs": lambda root, max_chars: load_fbs(root / "data/raw/fbs_sms_dataset_exported", max_chars),
    "cold": lambda root, max_chars: load_cold(root / "data/raw/cold", max_chars),
    "toxicn": lambda root, max_chars: load_toxicn(root / "data/raw/toxicn/ToxiCN_1.0.csv", max_chars),
    "chinese_offensive": lambda root, max_chars: load_chinese_offensive(root / "data/raw/chinese_offensive", max_chars),
}


def _balanced_take(records: list[RiskRecord], n: int, rng: random.Random) -> list[RiskRecord]:
    by_label: dict[int, list[RiskRecord]] = defaultdict(list)
    for record in records:
        label = record.labels.risk_binary
        if label in (0, 1):
            by_label[int(label)].append(record)

    if not by_label[0] or not by_label[1]:
        pool = list(records)
        rng.shuffle(pool)
        return pool[: min(n, len(pool))]

    half = n // 2
    n_pos = min(half, len(by_label[1]))
    n_neg = min(n - n_pos, len(by_label[0]))
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    selected = by_label[0][:n_neg] + by_label[1][:n_pos]
    if len(selected) < n:
        chosen_ids = {r.id for r in selected}
        rest = [r for r in records if r.id not in chosen_ids]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    rng.shuffle(selected)
    return selected


def _category_balanced_take(
    records: list[RiskRecord],
    n: int,
    rng: random.Random,
    max_fraction_per_category: float = 1.0,
) -> list[RiskRecord]:
    by_category: dict[str, list[RiskRecord]] = defaultdict(list)
    for record in records:
        by_category[record.labels.primary_category or "null"].append(record)

    categories = sorted(by_category)
    if not categories:
        return []

    quota = max(1, n // len(categories))
    selected: list[RiskRecord] = []
    for category in categories:
        pool = by_category[category]
        rng.shuffle(pool)
        cap = max(1, int(len(pool) * max_fraction_per_category))
        selected.extend(pool[: min(quota, cap, len(pool))])

    if len(selected) < n:
        chosen_ids = {r.id for r in selected}
        rest = [r for r in records if r.id not in chosen_ids]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def _proportional_counts(total: int, requested: list[int]) -> list[int]:
    requested_total = sum(requested)
    if total <= 0 or requested_total <= 0:
        return [0 for _ in requested]

    raw_counts = [total * value / requested_total for value in requested]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    order = sorted(
        range(len(requested)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _allocate_stratified_binary(
    records: list[RiskRecord],
    split_allocations: dict[str, int],
    split_order: list[str],
    rng: random.Random,
) -> dict[str, list[RiskRecord]]:
    by_label: dict[int, list[RiskRecord]] = defaultdict(list)
    for record in records:
        label = record.labels.risk_binary
        if label in (0, 1):
            by_label[int(label)].append(record)

    selected_by_split = {split: [] for split in split_order}
    requested = [int(split_allocations.get(split, 0)) for split in split_order]
    requested_total = sum(requested)

    for label_records in by_label.values():
        rng.shuffle(label_records)
        take_total = min(len(label_records), requested_total)
        counts = _proportional_counts(take_total, requested)
        offset = 0
        for split, count in zip(split_order, counts):
            selected_by_split[split].extend(label_records[offset : offset + count])
            offset += count

    for split in split_order:
        rng.shuffle(selected_by_split[split])
    return selected_by_split


def _take(
    records: list[RiskRecord],
    n: int,
    rng: random.Random,
    balanced_binary: bool,
    balanced_category: bool,
    split_name: str,
) -> list[RiskRecord]:
    if n <= 0:
        return []
    if balanced_category:
        max_fraction = {"train": 0.70, "dev": 0.50}.get(split_name, 1.0)
        return _category_balanced_take(records, n, rng, max_fraction_per_category=max_fraction)
    if balanced_binary:
        return _balanced_take(records, n, rng)
    pool = list(records)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def _summarize(records: list[RiskRecord]) -> dict:
    source_counts = Counter(r.metadata.get("source", "unknown") for r in records)
    binary_counts = Counter(str(r.labels.risk_binary) for r in records)
    category_counts = Counter(r.labels.primary_category or "null" for r in records)
    return {
        "records": len(records),
        "by_source": dict(sorted(source_counts.items())),
        "by_risk_binary": dict(sorted(binary_counts.items())),
        "by_primary_category": dict(sorted(category_counts.items())),
    }


def build_dataset(root: Path, profile_name: str) -> dict:
    config_path = root / "configs/dataset_profiles.json"
    profiles = json.loads(config_path.read_text(encoding="utf-8"))
    profile = profiles[profile_name]
    seed = int(profile.get("random_seed", 42))
    max_chars = int(profile.get("max_text_chars", 512))
    balanced_sources = set(profile.get("balanced_binary_sources", []))
    category_balanced_sources = set(profile.get("balanced_category_sources", []))
    stratified_binary_sources = set(profile.get("stratified_binary_sources", []))

    loaded: dict[str, list[RiskRecord]] = {}
    for source in sorted({s for split in profile["splits"].values() for s in split}):
        loaded[source] = LOADERS[source](root, max_chars)

    used_ids: set[str] = set()
    output_dir = root / "data/processed" / profile_name
    split_order = list(profile["splits"].keys())
    preallocated: dict[str, dict[str, list[RiskRecord]]] = {}
    for source in stratified_binary_sources:
        if source not in loaded:
            continue
        source_allocations = {
            split: int(profile["splits"][split].get(source, 0))
            for split in split_order
        }
        preallocated[source] = _allocate_stratified_binary(
            loaded[source],
            source_allocations,
            split_order,
            random.Random(seed + 1009),
        )

    manifest = {
        "profile": profile_name,
        "description": profile.get("description", ""),
        "loaded": {name: _summarize(records) for name, records in loaded.items()},
        "splits": {},
    }

    for split_name, allocation in profile["splits"].items():
        split_rng = random.Random(seed + {"train": 0, "dev": 1, "test": 2}.get(split_name, 9))
        split_records: list[RiskRecord] = []
        for source, n in allocation.items():
            if source in preallocated:
                selected = list(preallocated[source].get(split_name, []))
            else:
                candidates = [r for r in loaded[source] if r.id not in used_ids]
                selected = _take(
                    candidates,
                    int(n),
                    split_rng,
                    balanced_binary=source in balanced_sources,
                    balanced_category=source in category_balanced_sources,
                    split_name=split_name,
                )
            for record in selected:
                record.metadata["split"] = split_name
            used_ids.update(r.id for r in selected)
            split_records.extend(selected)
        split_rng.shuffle(split_records)
        out_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, split_records)
        manifest["splits"][split_name] = _summarize(split_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    (root / "data/manifests" / f"{profile_name}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--profile", default="quick20k")
    args = parser.parse_args()
    manifest = build_dataset(Path(args.root).resolve(), args.profile)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
