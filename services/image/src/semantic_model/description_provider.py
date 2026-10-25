"""
文件功能：图片描述生成器。通过 OpenAI 兼容 API 生成图片的
自然语言描述文本，作为语义模型的辅助输入。支持占位实现
（返回空描述）以便在不使用外部 API 时仍可运行。

核心类：
- OpenAICompatibleDescriptionProvider: 基于 API 的图片描述生成器
- EmptyDescriptionProvider: 占位描述生成器

核心函数：
- build_description_provider: 根据环境变量创建合适的描述生成器
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any

import requests


@dataclass(frozen=True)
class DescriptionResult:
    text_description: str = ""
    meme_description: str = ""
    provider: str = "empty"
    error: str | None = None
    raw_response: str | None = None


class EmptyDescriptionProvider:
    def describe(self, image_path: str | Path, text: str = "") -> DescriptionResult:
        return DescriptionResult()


def _read_config_value(config: dict[str, Any], key: str, env_key: str | None = None) -> str:
    value = str(config.get(key) or "").strip()
    if value:
        return value
    env_name = str(config.get(env_key or f"{key}_env") or "").strip()
    if env_name:
        return str(os.environ.get(env_name, "")).strip()
    return ""


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class OpenAICompatibleDescriptionProvider:
    """Generate ToxiCN_MM-style description fields with a vision chat model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.endpoint_url = _read_config_value(config, "endpoint_url")
        self.base_url = _read_config_value(config, "base_url")
        self.api_key = _read_config_value(config, "api_key", "api_key_env")
        self.model = _read_config_value(config, "model", "model_env")
        self.timeout_seconds = float(config.get("timeout_seconds", 45))
        self.max_tokens = int(config.get("max_tokens", 320))
        self.temperature = float(config.get("temperature", 0.1))
        self.provider_name = str(config.get("provider_name", "openai_compatible_vlm"))
        self.system_prompt = str(
            config.get(
                "system_prompt",
                (
                    "你是中文内容安全场景下的图片语义描述器。"
                    "只生成结构化描述，不做最终封禁判定，不输出额外解释。"
                ),
            )
        )
        self.user_prompt = str(
            config.get(
                "user_prompt",
                (
                    "请结合图片和OCR文字，输出严格JSON："
                    '{"text_description":"图中文字的含义、语气、暗示或攻击对象",'
                    '"meme_description":"整张图表达的梗、讽刺、冒犯、低俗、广告或潜在风险语义"}。'
                    "如果看不出风险也要客观描述，不要空字段。OCR文字：{text}"
                ),
            )
        )

    def _chat_endpoint(self) -> str:
        if self.endpoint_url:
            return self.endpoint_url
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def describe(self, image_path: str | Path, text: str = "") -> DescriptionResult:
        if not self.enabled:
            return DescriptionResult(provider=self.provider_name)
        if not self.model:
            return DescriptionResult(provider=self.provider_name, error="missing_model")
        if not (self.endpoint_url or self.base_url):
            return DescriptionResult(provider=self.provider_name, error="missing_endpoint")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.user_prompt.replace("{text}", text or "")},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = requests.post(
                self._chat_endpoint(),
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = str(data["choices"][0]["message"]["content"]).strip()
            parsed = _extract_json_object(content)
            return DescriptionResult(
                text_description=str(parsed.get("text_description") or "").strip(),
                meme_description=str(parsed.get("meme_description") or "").strip(),
                provider=self.provider_name,
                raw_response=content,
            )
        except Exception as exc:
            return DescriptionResult(provider=self.provider_name, error=str(exc))


def build_description_provider(config: dict[str, Any] | None) -> Any:
    provider_cfg = config or {}
    provider_type = str(provider_cfg.get("type", "empty")).strip().lower()
    if provider_type in {"openai_compatible", "openai-compatible", "vlm"}:
        return OpenAICompatibleDescriptionProvider(provider_cfg)
    return EmptyDescriptionProvider()
