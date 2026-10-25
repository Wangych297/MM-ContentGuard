"""
文件功能：URL 检测器。使用正则表达式检测 OCR 提取文本中的
URL 链接和域名信息。

核心函数：
- detect_urls: 检测文本中的 URL 和域名
"""

from __future__ import annotations

import re

from common.schemas import RuleHit


URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+|\b[A-Za-z0-9.-]+\.(?:com|cn|net|org|top|xyz|vip|cc|io)\b"
)


def detect_urls(text: str) -> list[RuleHit]:
    return [RuleHit(name="url", value=match.group(0)) for match in URL_PATTERN.finditer(text)]


