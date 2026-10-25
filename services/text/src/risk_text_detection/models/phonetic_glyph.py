"""
文件功能：音形融合特征编码器。将拼音特征和字形（glyph）特征融合为
统一的音形向量表示，作为多任务模型的辅助输入，增强对谐音变体
和形近字的检测能力。

核心类：
- PhoneticGlyphFeatureExtractor: 音形特征提取器
- PhoneticGlyphConfig: 音形编码器配置
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pypinyin import Style, lazy_pinyin

from risk_text_detection.augment.char_similarity import CharSimilarityIndex


INITIALS = [
    "",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "zh",
    "ch",
    "sh",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
]

FINALS = [
    "",
    "a",
    "o",
    "e",
    "i",
    "u",
    "v",
    "ai",
    "ei",
    "ui",
    "ao",
    "ou",
    "iu",
    "ie",
    "ve",
    "er",
    "an",
    "en",
    "in",
    "un",
    "vn",
    "ang",
    "eng",
    "ing",
    "ong",
    "ia",
    "iao",
    "ian",
    "iang",
    "iong",
    "ua",
    "uo",
    "uai",
    "uan",
    "uang",
    "ue",
]


def stable_hash_id(value: str, buckets: int, offset: int = 2) -> int:
    if not value:
        return 0
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16) % buckets + offset


def base36_id(char: str, buckets: int = 64) -> int:
    if not char:
        return 0
    try:
        return int(char, 36) % buckets + 2
    except ValueError:
        return stable_hash_id(char, buckets)


def is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


@dataclass(frozen=True)
class PhoneticGlyphVocabSizes:
    initial: int = len(INITIALS) + 2
    final: int = len(FINALS) + 2
    tone: int = 7
    pinyin: int = 514
    sound_code: int = 514
    structure: int = 66
    shape_code: int = 514
    stroke: int = 66
    char: int = 8194


class PhoneticGlyphFeatureExtractor:
    def __init__(
        self,
        index_file: str | Path = "data/raw/ast/hanzi.txt",
        neighbor_top_k: int = 4,
        min_neighbor_score: float = 0.35,
    ) -> None:
        self.index = CharSimilarityIndex.from_hanzi_file(
            index_file,
            top_k=neighbor_top_k,
            min_score=min_neighbor_score,
        )
        self.neighbor_top_k = int(neighbor_top_k)
        self.initial_to_id = {value: idx + 2 for idx, value in enumerate(INITIALS)}
        self.final_to_id = {value: idx + 2 for idx, value in enumerate(FINALS)}
        self._feature_cache: dict[str, dict[str, Any]] = {}

    def _pinyin_parts(self, char: str) -> tuple[int, int, int, int]:
        if not is_cjk_char(char):
            return 0, 0, 0, 0
        initial = lazy_pinyin(char, style=Style.INITIALS, strict=False, errors="ignore")
        final = lazy_pinyin(char, style=Style.FINALS_TONE3, strict=False, errors="ignore")
        full = lazy_pinyin(char, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
        initial_text = initial[0] if initial else ""
        final_text = final[0] if final else ""
        full_text = full[0] if full else ""
        tone = 0
        match = re.search(r"([1-5])$", full_text)
        if match:
            tone = int(match.group(1))
        final_text = re.sub(r"[1-5]$", "", final_text).replace("ü", "v")
        full_text = full_text.replace("ü", "v")
        return (
            self.initial_to_id.get(initial_text, 1),
            self.final_to_id.get(final_text, 1),
            tone + 1 if tone else 0,
            stable_hash_id(full_text, 512),
        )

    def _char_features(self, char: str) -> dict[str, Any]:
        cached = self._feature_cache.get(char)
        if cached is not None:
            return cached
        initial_id, final_id, tone_id, pinyin_id = self._pinyin_parts(char)
        code = self.index.char_code(char)
        sound_code = code.sound_code if code else ""
        shape_code = code.shape_code if code else ""
        neighbors = self.index.similar_chars(char)[: self.neighbor_top_k]
        neighbor_ids = [stable_hash_id(item.char, 8192) for item in neighbors]
        neighbor_scores = [float(item.score) for item in neighbors]
        pad = self.neighbor_top_k - len(neighbor_ids)
        if pad > 0:
            neighbor_ids.extend([0] * pad)
            neighbor_scores.extend([0.0] * pad)
        features = {
            "initial_ids": initial_id,
            "final_ids": final_id,
            "tone_ids": tone_id,
            "pinyin_ids": pinyin_id,
            "sound_code_ids": stable_hash_id(sound_code, 512),
            "structure_ids": base36_id(shape_code[:1]),
            "shape_code_ids": stable_hash_id(shape_code, 512),
            "stroke_ids": base36_id(shape_code[-1:]),
            "char_ids": stable_hash_id(char, 8192) if is_cjk_char(char) else 0,
            "neighbor_char_ids": neighbor_ids,
            "neighbor_scores": neighbor_scores,
        }
        self._feature_cache[char] = features
        return features

    def batch_encode(
        self,
        texts: list[str],
        offset_mapping: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = offset_mapping.shape
        feature_names = [
            "initial_ids",
            "final_ids",
            "tone_ids",
            "pinyin_ids",
            "sound_code_ids",
            "structure_ids",
            "shape_code_ids",
            "stroke_ids",
            "char_ids",
        ]
        encoded: dict[str, Any] = {
            name: torch.zeros((batch_size, seq_len), dtype=torch.long)
            for name in feature_names
        }
        encoded["neighbor_char_ids"] = torch.zeros(
            (batch_size, seq_len, self.neighbor_top_k),
            dtype=torch.long,
        )
        encoded["neighbor_scores"] = torch.zeros(
            (batch_size, seq_len, self.neighbor_top_k),
            dtype=torch.float,
        )

        for batch_idx, text in enumerate(texts):
            for token_idx in range(seq_len):
                start = int(offset_mapping[batch_idx, token_idx, 0])
                end = int(offset_mapping[batch_idx, token_idx, 1])
                if end <= start or start >= len(text):
                    continue
                token_text = text[start:end]
                char = token_text[0] if len(token_text) == 1 else ""
                if not is_cjk_char(char):
                    continue
                features = self._char_features(char)
                for name in feature_names:
                    encoded[name][batch_idx, token_idx] = int(features[name])
                encoded["neighbor_char_ids"][batch_idx, token_idx] = torch.tensor(
                    features["neighbor_char_ids"],
                    dtype=torch.long,
                )
                encoded["neighbor_scores"][batch_idx, token_idx] = torch.tensor(
                    features["neighbor_scores"],
                    dtype=torch.float,
                )
        return encoded


class PhoneticGlyphBranch(nn.Module):
    def __init__(
        self,
        embed_dim: int = 48,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        vocab_sizes: PhoneticGlyphVocabSizes | None = None,
    ) -> None:
        super().__init__()
        vocab_sizes = vocab_sizes or PhoneticGlyphVocabSizes()
        self.embeddings = nn.ModuleDict(
            {
                "initial_ids": nn.Embedding(vocab_sizes.initial, embed_dim, padding_idx=0),
                "final_ids": nn.Embedding(vocab_sizes.final, embed_dim, padding_idx=0),
                "tone_ids": nn.Embedding(vocab_sizes.tone, embed_dim, padding_idx=0),
                "pinyin_ids": nn.Embedding(vocab_sizes.pinyin, embed_dim, padding_idx=0),
                "sound_code_ids": nn.Embedding(vocab_sizes.sound_code, embed_dim, padding_idx=0),
                "structure_ids": nn.Embedding(vocab_sizes.structure, embed_dim, padding_idx=0),
                "shape_code_ids": nn.Embedding(vocab_sizes.shape_code, embed_dim, padding_idx=0),
                "stroke_ids": nn.Embedding(vocab_sizes.stroke, embed_dim, padding_idx=0),
                "char_ids": nn.Embedding(vocab_sizes.char, embed_dim, padding_idx=0),
            }
        )
        self.neighbor_embedding = nn.Embedding(vocab_sizes.char, embed_dim, padding_idx=0)
        self.input_projection = nn.Sequential(
            nn.Linear(embed_dim * (len(self.embeddings) + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        pieces = [embedding(features[name]) for name, embedding in self.embeddings.items()]
        neighbor_ids = features["neighbor_char_ids"]
        neighbor_scores = features["neighbor_scores"].unsqueeze(-1)
        neighbor_emb = self.neighbor_embedding(neighbor_ids)
        neighbor_feature = (neighbor_emb * neighbor_scores).sum(dim=2)
        pieces.append(neighbor_feature)
        x = self.input_projection(torch.cat(pieces, dim=-1))
        contextual = self.context(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + contextual)
