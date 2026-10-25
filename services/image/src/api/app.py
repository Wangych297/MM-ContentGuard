"""
文件功能：图片审核 FastAPI 服务。提供 /image/check 审核端点和 /health 健康检查，
在启动时预加载图片审核流水线（OCR、检测器、视觉模型、语义模型），
支持模型热更新和临时文件清理。

核心接口：
- POST /image/check: 接收图片路径，返回审核结果
- GET /health: 服务健康检查
"""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pipeline.pipeline import ImageRiskPipeline

DEFAULT_IMAGE_CONFIG = "configs/image_classifier_dinov3_large.yaml"
DEFAULT_IMAGE_CHECKPOINT = Path("checkpoints/image_model_dinov3_large/best.pt")
DEFAULT_SEMANTIC_CONFIG = "configs/semantic_meme_focal_binary.yaml"
DEFAULT_SEMANTIC_CHECKPOINT = Path("checkpoints/semantic_model_focal_binary/best.pt")


class ImageCheckRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    use_ocr: bool = True


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _warmup_image_path() -> Path:
    configured = os.getenv("RISK_IMAGE_WARMUP_IMAGE")
    if configured:
        return Path(configured)

    warmup_dir = Path(os.getenv("RISK_IMAGE_WARMUP_DIR") or tempfile.gettempdir())
    warmup_dir.mkdir(parents=True, exist_ok=True)
    image_path = warmup_dir / "risk_image_warmup.jpg"
    if not image_path.exists():
        from PIL import Image

        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(image_path)
    return image_path


@lru_cache(maxsize=1)
def get_image_pipeline() -> ImageRiskPipeline:
    return ImageRiskPipeline(config_path=DEFAULT_IMAGE_CONFIG, load_visual=True)


app = FastAPI(title="Risk Image Detection API", version="0.1.0")


@app.on_event("startup")
def preload_image_pipeline() -> None:
    if not _env_flag("RISK_IMAGE_PRELOAD", True):
        return

    started_at = time.perf_counter()
    pipeline = get_image_pipeline()
    warmup_status = "pipeline_only"

    if _env_flag("RISK_IMAGE_WARMUP_INFERENCE", True):
        warmup_image = _warmup_image_path()
        warmup_status = f"inference:{warmup_image}"
        try:
            pipeline.predict(
                warmup_image,
                use_ocr=_env_flag("RISK_IMAGE_WARMUP_OCR", True),
            )
        except Exception as exc:
            warmup_status = f"inference_failed:{exc}"

    elapsed = time.perf_counter() - started_at
    print(
        "Risk image pipeline preloaded in %.2fs (%s)"
        % (elapsed, warmup_status),
        flush=True,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "image_config": DEFAULT_IMAGE_CONFIG,
        "image_checkpoint": str(DEFAULT_IMAGE_CHECKPOINT),
        "image_checkpoint_exists": DEFAULT_IMAGE_CHECKPOINT.exists(),
        "semantic_config": DEFAULT_SEMANTIC_CONFIG,
        "semantic_checkpoint": str(DEFAULT_SEMANTIC_CHECKPOINT),
        "semantic_checkpoint_exists": DEFAULT_SEMANTIC_CHECKPOINT.exists(),
    }


@app.post("/image/check")
def check_image(request: ImageCheckRequest) -> dict[str, Any]:
    pipeline = get_image_pipeline()
    return pipeline.predict(request.image_path, use_ocr=request.use_ocr)


def main() -> None:
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8010, reload=False)


if __name__ == "__main__":
    main()



