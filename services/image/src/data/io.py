"""
文件功能：数据 I/O 工具。提供 JSONL（JSON Lines）格式文件的
读写功能，用于图片数据集的记录存取。

核心函数：
- read_jsonl: 逐行读取 JSONL 文件，返回字典迭代器
- write_jsonl: 将字典列表写入 JSONL 文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count

