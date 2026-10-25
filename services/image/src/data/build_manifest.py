"""
文件功能：图片清单构建。扫描 SIMAS/PASS 两个图片数据集目录，
按照标签配置生成包含标签信息的 ImageRecord JSONL 清单文件。

核心函数：
- build_manifest: 扫描目录构建图片记录清单
- main: 命令行入口
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common.config import ensure_parent, load_yaml, resolve_path
from common.image_io import is_image_file
from common.schemas import ImageRecord
from data.io import write_jsonl


SIMAS_DIR_TO_LABEL = {
    "alcohol": "regulated_goods",
    "alcohol_safe": "safe",
    "drugs": "drug",
    "drugs_safe": "safe",
    "firearms": "weapon",
    "firearms_safe": "safe",
    "gambling": "gambling",
    "gambling_safe": "safe",
    "sexual": "sexual",
    "sexual_safe": "safe",
    "tobacco": "regulated_goods",
    "tobacco_safe": "safe",
    "violence": "violence",
    "violence_safe": "safe",
}


def _iter_images(root: Path, allowed_extensions: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_image_file(path, allowed_extensions))


def _build_simas_records(root: Path, allowed_extensions: set[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not root.exists():
        raise FileNotFoundError(f"SIMAS root does not exist: {root}")

    for directory_name, label in SIMAS_DIR_TO_LABEL.items():
        directory = root / directory_name
        if not directory.exists():
            continue
        for idx, image_path in enumerate(_iter_images(directory, allowed_extensions), start=1):
            records.append(
                ImageRecord(
                    id=f"simas_{directory_name}_{idx:05d}",
                    image_path=str(image_path.resolve()),
                    label=label,
                    coarse_label="safe" if label == "safe" else "unsafe",
                    is_safe=label == "safe",
                    source="simas",
                    original_label=directory_name,
                    metadata={"source_dir": directory_name},
                )
            )
    return records


def _build_pass_records(root: Path, allowed_extensions: set[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not root.exists():
        return records

    for idx, image_path in enumerate(_iter_images(root, allowed_extensions), start=1):
        records.append(
            ImageRecord(
                id=f"pass_safe_{idx:05d}",
                image_path=str(image_path.resolve()),
                label="safe",
                coarse_label="safe",
                is_safe=True,
                source="pass",
                original_label="pass_safe",
                metadata={"source_dir": str(root.resolve())},
            )
        )
    return records


def build_manifest(config: dict, output_path: str | Path | None = None) -> list[ImageRecord]:
    data_cfg = config.get("data", {})
    allowed_extensions = {ext.lower() for ext in data_cfg.get("allowed_extensions", [])}
    if not allowed_extensions:
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    simas_root = resolve_path(data_cfg.get("simas_root", "../simas"))
    pass_root = resolve_path(data_cfg.get("pass_safe_root", "../safe_extra/pass_1500/images"))
    manifest_path = ensure_parent(output_path or data_cfg.get("manifest_path", "data/processed/image_manifest.jsonl"))

    records = _build_simas_records(simas_root, allowed_extensions)
    records.extend(_build_pass_records(pass_root, allowed_extensions))

    count = write_jsonl(manifest_path, [record.to_dict() for record in records])
    by_label = Counter(record.label for record in records)
    by_source = Counter(record.source for record in records)

    print(f"Wrote {count} records to {manifest_path}")
    print("By label:", dict(sorted(by_label.items())))
    print("By source:", dict(sorted(by_source.items())))
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_classifier_dinov3_large.yaml")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    build_manifest(config, output_path=args.output)


if __name__ == "__main__":
    main()



