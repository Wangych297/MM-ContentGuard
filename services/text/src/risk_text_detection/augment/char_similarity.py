"""
文件功能：字符相似度计算。基于编辑距离和音形特征计算两个中文字符
之间的相似度分数，用于数据增强中的字符替换策略。

核心函数：
- char_similarity: 计算两个字符的综合相似度
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from risk_text_detection.augment.char_variant import CharCode, CharVariantIndex, is_cjk_char


SOUND_WEIGHTS = (0.35, 0.35, 0.15, 0.15)
SHAPE_WEIGHTS = (0.18, 0.16, 0.14, 0.14, 0.14, 0.14, 0.10)


@dataclass(frozen=True)
class SimilarChar:
    char: str
    score: float
    relation: str
    sound_score: float
    shape_score: float


def weighted_code_similarity(left: str, right: str, weights: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    total = 0.0
    for idx, weight in enumerate(weights):
        if idx < len(left) and idx < len(right) and left[idx] == right[idx]:
            total += weight
    return total


def char_similarity(
    left: CharCode,
    right: CharCode,
    sound_weight: float = 0.55,
    shape_weight: float = 0.45,
) -> SimilarChar:
    sound_score = weighted_code_similarity(left.sound_code, right.sound_code, SOUND_WEIGHTS)
    shape_score = weighted_code_similarity(left.shape_code, right.shape_code, SHAPE_WEIGHTS)
    score = sound_weight * sound_score + shape_weight * shape_score
    if sound_score >= 0.7 and shape_score >= 0.45:
        relation = "mixed"
    elif sound_score >= shape_score:
        relation = "homophone"
    else:
        relation = "glyph"
    return SimilarChar(
        char=right.char,
        score=float(score),
        relation=relation,
        sound_score=float(sound_score),
        shape_score=float(shape_score),
    )


class CharSimilarityIndex:
    def __init__(self, base_index: CharVariantIndex, neighbors: dict[str, list[SimilarChar]]) -> None:
        self.base_index = base_index
        self.neighbors = neighbors

    @classmethod
    def from_hanzi_file(
        cls,
        path: str | Path,
        min_count: int = 1,
        top_k: int = 8,
        min_score: float = 0.35,
    ) -> "CharSimilarityIndex":
        base = CharVariantIndex.from_hanzi_file(path, min_count=min_count)
        by_sound: dict[str, list[CharCode]] = defaultdict(list)
        by_shape: dict[str, list[CharCode]] = defaultdict(list)
        for entry in base.entries:
            by_sound[entry.sound_code].append(entry)
            by_shape[entry.coarse_shape_code].append(entry)

        neighbors: dict[str, list[SimilarChar]] = {}
        for entry in base.entries:
            candidate_entries: dict[str, CharCode] = {}
            for candidate in by_sound.get(entry.sound_code, []):
                if candidate.char != entry.char:
                    candidate_entries[candidate.char] = candidate
            for candidate in by_shape.get(entry.coarse_shape_code, []):
                if candidate.char != entry.char:
                    candidate_entries[candidate.char] = candidate

            scored = [
                char_similarity(entry, candidate)
                for candidate in candidate_entries.values()
                if is_cjk_char(candidate.char)
            ]
            filtered = [item for item in scored if item.score >= min_score]
            neighbors[entry.char] = sorted(filtered, key=lambda item: item.score, reverse=True)[:top_k]
        return cls(base, neighbors)

    def char_code(self, char: str) -> CharCode | None:
        return self.base_index.char_to_entry.get(char)

    def similar_chars(self, char: str) -> list[SimilarChar]:
        return self.neighbors.get(char, [])
