"""
文件功能：文本翻译适配器。图片审核流水线中的翻译环节适配器，
当前为占位实现（直接返回原文），预留接口以便后续接入
真实的翻译服务。

核心类：
- TextTranslator: 文本翻译适配器
"""

from __future__ import annotations


class TextTranslator:
    """Placeholder translator adapter.

    The image pipeline keeps translation outside the visual model. This adapter currently
    returns text unchanged, so a real translator can be plugged in later without
    changing the cascade pipeline.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def translate_to_chinese(self, text: str) -> dict:
        return {
            "enabled": self.enabled,
            "source_text": text,
            "translated_text": text,
            "engine": "identity",
        }
