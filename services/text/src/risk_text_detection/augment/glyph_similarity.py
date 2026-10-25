"""
文件功能：字形相似度计算。使用字体渲染计算两个字符的视觉字形
（glyph）相似度，用于生成形近字替换候选。

核心类：
- GlyphNeighbor: 字形邻居数据结构

核心函数：
- glyph_similarity: 计算两个字符的字形相似度分数
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlyphNeighbor:
    char: str
    score: float
    relation: str


class GlyphSimilarityIndex:
    def __init__(self, neighbors: dict[str, list[GlyphNeighbor]]) -> None:
        self.neighbors = neighbors

    @classmethod
    def from_edge_file(
        cls,
        path: str | Path,
        min_score: float = 0.75,
        max_neighbors_per_char: int = 20,
    ) -> "GlyphSimilarityIndex":
        path = Path(path)
        neighbors: dict[str, list[GlyphNeighbor]] = defaultdict(list)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                left, right = parts[0], parts[1]
                try:
                    score = float(parts[2])
                except ValueError:
                    continue
                if len(left) != 1 or len(right) != 1 or score < min_score:
                    continue
                relation = parts[3] if len(parts) > 3 else "glyph"
                neighbors[left].append(GlyphNeighbor(right, score, relation))
                neighbors[right].append(GlyphNeighbor(left, score, relation))

        trimmed = {
            char: sorted(items, key=lambda item: item.score, reverse=True)[:max_neighbors_per_char]
            for char, items in neighbors.items()
        }
        return cls(trimmed)

    def candidates(self, char: str) -> list[GlyphNeighbor]:
        return self.neighbors.get(char, [])

    def replace_first_available(self, text: str) -> str | None:
        chars = list(text)
        for idx, char in enumerate(chars):
            candidates = self.candidates(char)
            if candidates:
                chars[idx] = candidates[0].char
                return "".join(chars)
        return None

