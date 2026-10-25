"""
文件功能：网关配置管理。从环境变量加载运行时配置（服务地址、超时时间、
审核阈值、文件大小限制等），封装为 GatewaySettings dataclass。

核心类：
- GatewaySettings: 网关配置数据类，提供 from_env() 工厂方法从环境变量构建

核心函数：
- get_settings: 获取网关配置的单例入口
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class GatewaySettings:
    storage_root: Path = Path("storage")
    cors_origins: list[str] = None
    text_api_backend: str = "risk_text"
    text_api_url: str = "http://127.0.0.1:8000"
    image_api_url: str = "http://127.0.0.1:8010"
    video_api_url: str = "http://127.0.0.1:8020"
    text_api_timeout: float = 15.0
    image_api_timeout: float = 120.0
    video_api_timeout: float = 300.0
    text_risk_threshold: float = 0.5
    text_suspicious_threshold: float = 0.5
    video_threshold: float = 0.32
    image_use_ocr: bool = True
    max_image_mb: int = 10
    max_video_mb: int = 200

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            storage_root=Path(os.getenv("GATEWAY_STORAGE_ROOT", str(cls.storage_root))),
            cors_origins=_env_list(
                "GATEWAY_CORS_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ),
            text_api_backend=os.getenv("TEXT_API_BACKEND", cls.text_api_backend).strip().lower(),
            text_api_url=os.getenv("TEXT_API_URL", cls.text_api_url).rstrip("/"),
            image_api_url=os.getenv("IMAGE_API_URL", cls.image_api_url).rstrip("/"),
            video_api_url=os.getenv("VIDEO_API_URL", cls.video_api_url).rstrip("/"),
            text_api_timeout=_env_float("TEXT_API_TIMEOUT", cls.text_api_timeout),
            image_api_timeout=_env_float("IMAGE_API_TIMEOUT", cls.image_api_timeout),
            video_api_timeout=_env_float("VIDEO_API_TIMEOUT", cls.video_api_timeout),
            text_risk_threshold=_env_float("TEXT_RISK_THRESHOLD", cls.text_risk_threshold),
            text_suspicious_threshold=_env_float(
                "TEXT_SUSPICIOUS_THRESHOLD",
                cls.text_suspicious_threshold,
            ),
            video_threshold=_env_float("VIDEO_THRESHOLD", cls.video_threshold),
            image_use_ocr=_env_bool("IMAGE_USE_OCR", cls.image_use_ocr),
            max_image_mb=_env_int("MAX_IMAGE_MB", cls.max_image_mb),
            max_video_mb=_env_int("MAX_VIDEO_MB", cls.max_video_mb),
        )


def get_settings() -> GatewaySettings:
    return GatewaySettings.from_env()
