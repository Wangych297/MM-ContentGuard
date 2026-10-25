"""
文件功能：文本审核 FastAPI 服务。提供 /text/check 审核端点和 /health 健康检查，
根据环境变量加载 BERT 多任务模型检查点并管理模型生命周期。

核心接口：
- POST /text/check: 接收文本，返回风险等级和标签
- GET /health: 服务健康检查
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from risk_text_detection.inference.tiered_review import (
    DEFAULT_GENERAL_CHECKPOINT_DIR,
    DEFAULT_REASON_CHECKPOINT_DIR,
    DEFAULT_SOCIAL_CHECKPOINT_DIR,
    TieredReviewConfig,
    TieredTextReviewPredictor,
)


class TextCheckRequest(BaseModel):
    text: str = Field(..., min_length=1)
    risk_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicious_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchTextCheckRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    risk_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicious_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _base_config() -> TieredReviewConfig:
    return TieredReviewConfig(
        general_checkpoint_dir=os.getenv(
            "RISK_TEXT_GENERAL_CHECKPOINT",
            os.getenv("RISK_TEXT_CHECKPOINT", DEFAULT_GENERAL_CHECKPOINT_DIR),
        ),
        social_checkpoint_dir=os.getenv(
            "RISK_TEXT_SOCIAL_CHECKPOINT",
            DEFAULT_SOCIAL_CHECKPOINT_DIR,
        ),
        reason_checkpoint_dir=os.getenv(
            "RISK_TEXT_REASON_CHECKPOINT",
            DEFAULT_REASON_CHECKPOINT_DIR,
        ),
        device=os.getenv("RISK_TEXT_DEVICE") or None,
        max_length=int(os.getenv("RISK_TEXT_MAX_LENGTH", "128")),
        general_threshold=_env_float(
            "RISK_TEXT_GENERAL_THRESHOLD",
            _env_float("RISK_TEXT_RISK_THRESHOLD", 0.5),
        ),
        social_review_threshold=_env_float(
            "RISK_TEXT_SOCIAL_REVIEW_THRESHOLD",
            _env_float("RISK_TEXT_SUSPICIOUS_THRESHOLD", 0.5),
        ),
        enable_reason=_env_flag("RISK_TEXT_ENABLE_REASON", True),
        parallel_binary=_env_flag("RISK_TEXT_PARALLEL_BINARY", True),
        parallel_reason=_env_flag("RISK_TEXT_PARALLEL_REASON", True),
    )


@lru_cache(maxsize=1)
def get_predictor() -> TieredTextReviewPredictor:
    return TieredTextReviewPredictor(_base_config())


app = FastAPI(title="Risk Text Detection API", version="0.1.0")


@app.on_event("startup")
def preload_predictor() -> None:
    if not _env_flag("RISK_TEXT_PRELOAD", True):
        return

    started_at = time.perf_counter()
    predictor = get_predictor()

    if _env_flag("RISK_TEXT_WARMUP_INFERENCE", True):
        predictor.predict(os.getenv("RISK_TEXT_WARMUP_TEXT", "文本审核预热"))

    elapsed = time.perf_counter() - started_at
    print(
        "Risk text predictor preloaded in %.2fs on %s"
        % (elapsed, str(predictor.device)),
        flush=True,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    config = _base_config()
    return {
        "status": "ok",
        "policy": "tiered_review",
        "checkpoint_dir": str(config.general_checkpoint_dir),
        "general_checkpoint_dir": str(config.general_checkpoint_dir),
        "social_checkpoint_dir": str(config.social_checkpoint_dir),
        "reason_checkpoint_dir": str(config.reason_checkpoint_dir),
        "general_threshold": config.general_threshold,
        "social_review_threshold": config.social_review_threshold,
        "enable_reason": config.enable_reason,
        "parallel_binary": config.parallel_binary,
        "parallel_reason": config.parallel_reason,
        "device": config.device or "auto",
    }


@app.post("/text/check")
def check_text(request: TextCheckRequest) -> dict[str, Any]:
    predictor = get_predictor()
    return predictor.predict(
        request.text,
        risk_threshold=request.risk_threshold,
        suspicious_threshold=request.suspicious_threshold,
    )


@app.post("/text/batch-check")
def check_text_batch(request: BatchTextCheckRequest) -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "results": predictor.predict_batch(
            request.texts,
            risk_threshold=request.risk_threshold,
            suspicious_threshold=request.suspicious_threshold,
        )
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "risk_text_detection.api.app:app",
        host=os.getenv("RISK_TEXT_HOST", "127.0.0.1"),
        port=int(os.getenv("RISK_TEXT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
