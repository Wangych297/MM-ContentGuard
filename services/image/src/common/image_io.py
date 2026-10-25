"""
文件功能：图片 I/O 工具。提供 RGB 图片加载和图片文件类型判断
功能，使用 PIL 库处理常见图片格式。

核心函数：
- load_rgb_image: 从磁盘加载图片并转换为 RGB 模式的 PIL Image
- is_image_file: 根据文件扩展名判断是否为支持的图片格式
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def is_image_file(path: Path, allowed_extensions: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_extensions

