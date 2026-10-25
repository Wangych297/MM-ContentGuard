"""
文件功能：拼音编码器。将中文文本转换为拼音序列，通过可训练的
嵌入层和编码器学习拼音级别的语义表示，用于辅助检测同音替换等
规避手段。

核心函数：
- text_to_pinyin: 将中文文本转换为拼音字符串

核心类：
- PinyinEncoder: 拼音序列编码器
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pypinyin import Style, lazy_pinyin


def text_to_pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore")).lower()


class PinyinCharTokenizer:
    def __init__(self) -> None:
        self.pad_id = 0
        self.unk_id = 1
        self._char_to_id: dict[str, int] = {" ": 2}
        idx = 3
        for char in "abcdefghijklmnopqrstuvwxyz":
            self._char_to_id[char] = idx
            idx += 1
        for char in "0123456789":
            self._char_to_id[char] = idx
            idx += 1
        self._vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return [self._char_to_id.get(char, self.unk_id) for char in text]

    def batch_encode(self, texts: list[str], max_length: int) -> torch.Tensor:
        batch: list[list[int]] = []
        for text in texts:
            ids = self.encode(text)[:max_length]
            ids.extend([self.pad_id] * (max_length - len(ids)))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)


class PinyinCNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.pointwise = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings.permute(0, 2, 1)
        x = self.depthwise(x)
        x = F.relu(self.pointwise(x))
        return self.pool(x).squeeze(-1)


class PinyinBranch(nn.Module):
    def __init__(self, vocab_size: int = 39, embed_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = PinyinCNNEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, pinyin_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(pinyin_ids))

