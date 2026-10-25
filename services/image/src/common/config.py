"""
文件功能：配置工具函数。提供 YAML 配置文件加载、路径解析、
点号分隔键值访问和项目根目录定位等通用配置操作。

核心函数：
- load_yaml: 加载 YAML 配置文件
- resolve_path: 将相对路径解析为绝对路径
- config_get: 通过点号分隔的键路径获取嵌套配置值
- project_root: 返回项目根目录路径
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    base_path = Path(base) if base is not None else project_root()
    return (base_path / value).resolve()


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def config_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
