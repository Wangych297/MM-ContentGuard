"""
文件功能：联系方式检测器。使用正则表达式检测文本中的电话号码、
电子邮箱和 QQ/社交媒体账号等联系方式信息。

核心函数：
- detect_contacts: 检测文本中的联系方式，返回 RuleHit 列表
"""

from __future__ import annotations

import re

from common.schemas import RuleHit


PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
QQ_PATTERN = re.compile(r"(?i)\b(?:qq|q)[\s:锛?]*([1-9]\d{4,11})\b")
WECHAT_PATTERN = re.compile(r"(?i)\b(?:wechat|weixin|wx|v淇寰甗淇)[\s:锛?]*([A-Za-z][-_A-Za-z0-9]{5,19})")


def detect_contacts(text: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for match in PHONE_PATTERN.finditer(text):
        hits.append(RuleHit(name="phone", value=match.group(0)))
    for match in EMAIL_PATTERN.finditer(text):
        hits.append(RuleHit(name="email", value=match.group(0)))
    for match in QQ_PATTERN.finditer(text):
        hits.append(RuleHit(name="qq", value=match.group(0)))
    for match in WECHAT_PATTERN.finditer(text):
        hits.append(RuleHit(name="wechat", value=match.group(0)))
    return hits


