"""
文件功能：字符变体生成。生成中文字符的变体列表，包括同音字替换、
形近字替换和跨书写系统替换，用于文本数据增强以提升模型鲁棒性。

核心类：
- CharVariantGenerator: 字符变体生成器，管理字符相似度索引和替换策略
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


@dataclass(frozen=True)
class CharCode:
    char: str
    code: str
    count: int

    @property
    def sound_code(self) -> str:
        return self.code[:4]

    @property
    def shape_code(self) -> str:
        return self.code[4:]

    @property
    def coarse_shape_code(self) -> str:
        if len(self.shape_code) < 2:
            return self.shape_code
        return f"{self.shape_code[:3]}:{self.shape_code[-1]}"


class CharVariantIndex:
    def __init__(self, entries: list[CharCode]) -> None:
        self.entries = entries
        self.char_to_entry = {entry.char: entry for entry in entries}
        self.by_sound = self._build_group(lambda entry: entry.sound_code)
        self.by_shape = self._build_group(lambda entry: entry.coarse_shape_code)

    @classmethod
    def from_hanzi_file(
        cls,
        path: str | Path,
        min_count: int = 1,
    ) -> "CharVariantIndex":
        entries: list[CharCode] = []
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                char, code = parts[0], parts[1]
                if len(char) != 1 or not is_cjk_char(char) or len(code) < 5:
                    continue
                try:
                    count = int(parts[2]) if len(parts) >= 3 else 1
                except ValueError:
                    count = 1
                if count >= min_count:
                    entries.append(CharCode(char=char, code=code, count=count))
        return cls(entries)

    def _build_group(self, key_fn) -> dict[str, list[str]]:
        grouped: dict[str, list[CharCode]] = defaultdict(list)
        for entry in self.entries:
            key = key_fn(entry)
            if key:
                grouped[key].append(entry)
        return {
            key: [entry.char for entry in sorted(items, key=lambda item: item.count, reverse=True)]
            for key, items in grouped.items()
            if len(items) > 1
        }

    def candidates(self, char: str, mode: str) -> list[str]:
        entry = self.char_to_entry.get(char)
        if entry is None:
            return []
        if mode == "homophone":
            pool = self.by_sound.get(entry.sound_code, [])
        elif mode == "glyph":
            pool = self.by_shape.get(entry.coarse_shape_code, [])
        else:
            sound_pool = self.by_sound.get(entry.sound_code, [])
            shape_pool = self.by_shape.get(entry.coarse_shape_code, [])
            pool = list(dict.fromkeys(sound_pool + shape_pool))
        return [candidate for candidate in pool if candidate != char]


class CharVariantAugmenter:
    def __init__(
        self,
        index: CharVariantIndex,
        mode: str = "mixed",
        replace_prob: float = 0.15,
        max_replacements: int = 2,
        seed: int = 42,
    ) -> None:
        if mode not in {"homophone", "glyph", "mixed"}:
            raise ValueError("mode must be one of: homophone, glyph, mixed")
        self.index = index
        self.mode = mode
        self.replace_prob = float(replace_prob)
        self.max_replacements = int(max_replacements)
        self.rng = random.Random(seed)

    def augment(self, text: str) -> str:
        if not text:
            return text
        chars = list(text)
        positions = [idx for idx, char in enumerate(chars) if self.index.candidates(char, self.mode)]
        if not positions:
            return text
        self.rng.shuffle(positions)
        changed = 0
        for idx in positions:
            if changed >= self.max_replacements:
                break
            if self.rng.random() > self.replace_prob:
                continue
            candidates = self.index.candidates(chars[idx], self.mode)
            if not candidates:
                continue
            chars[idx] = self.rng.choice(candidates)
            changed += 1
        return "".join(chars) if changed else text
