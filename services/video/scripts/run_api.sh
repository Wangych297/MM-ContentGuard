#!/usr/bin/env bash
#
# 文件功能：启动视频审核 API 服务。配置多模态模型路径和 GPU
# 设备后，通过 Uvicorn 启动 FastAPI 服务。
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn src.api.app:app --host "$HOST" --port "$PORT"
