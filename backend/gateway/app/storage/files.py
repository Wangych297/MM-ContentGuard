"""
文件功能：文件上传存储工具。处理用户上传附件的临时存储、类型检测、
文件名清理和文件移动等操作。

核心类：
- SavedUpload: 已保存上传文件的数据类，记录临时路径、原始名称和模态类型

核心函数：
- detect_modality: 根据文件扩展名和 MIME 类型判断附件模态（image/video）
- save_upload_to_temp: 将上传文件保存到临时目录
- move_upload: 将审核通过的文件移动到最终存储路径
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4"}


@dataclass(frozen=True)
class SavedUpload:
    temp_path: Path
    original_name: str
    mime_type: str
    size_bytes: int
    modality: str


def sanitize_filename(filename: str) -> str:
    safe = Path(filename or "upload").name
    return "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in safe)


def detect_modality(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        return "image"
    if suffix in VIDEO_EXTENSIONS or content_type == "video/mp4":
        return "video"
    raise HTTPException(
        status_code=400,
        detail="Unsupported attachment type. Use jpg/png/webp/bmp images or mp4 videos.",
    )


def save_upload_to_temp(upload: UploadFile, tmp_dir: Path) -> SavedUpload:
    original_name = sanitize_filename(upload.filename or "upload")
    mime_type = upload.content_type or "application/octet-stream"
    modality = detect_modality(original_name, mime_type)
    suffix = Path(original_name).suffix.lower()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tmp_dir / ("%s%s" % (uuid.uuid4().hex, suffix))
    with temp_path.open("wb") as output:
        _copy_fileobj(upload.file, output)
    size_bytes = temp_path.stat().st_size
    return SavedUpload(
        temp_path=temp_path,
        original_name=original_name,
        mime_type=mime_type,
        size_bytes=size_bytes,
        modality=modality,
    )


def move_upload(upload: SavedUpload, target_dir: Path, attachment_id: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = upload.temp_path.suffix
    target_path = target_dir / ("%s%s" % (attachment_id, suffix))
    shutil.move(str(upload.temp_path), target_path)
    return target_path


def _copy_fileobj(source: BinaryIO, target: BinaryIO) -> None:
    shutil.copyfileobj(source, target, length=1024 * 1024)
