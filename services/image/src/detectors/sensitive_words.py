"""
文件功能：敏感词检测器。检查文本是否命中预设的敏感词/黑名单
词汇列表，适用于 OCR 提取后的文本审核。

核心函数：
- detect_sensitive_words: 在文本中检测配置的敏感词
"""

from __future__ import annotations

from common.schemas import RuleHit


DEFAULT_SENSITIVE_WORDS = [
    "gambling",
    "casino",
    "betting",
    "loan",
    "payday",
    "escort",
    "porn",
    "drug",
    "pills",
    "gun",
    "weapon",
    "\u535a\u5f69",
    "\u8d4c\u535a",
    "\u8d37\u6b3e",
    "\u88f8\u804a",
    "\u8272\u60c5",
    "\u6bd2\u54c1",
    "\u67aa",
    "\u529e\u8bc1",
]


def detect_sensitive_words(text: str, words: list[str] | None = None) -> list[RuleHit]:
    lowered = text.lower()
    hits: list[RuleHit] = []
    for word in words or DEFAULT_SENSITIVE_WORDS:
        needle = word.lower()
        if needle and needle in lowered:
            hits.append(RuleHit(name="sensitive_word", value=word))
    return hits


