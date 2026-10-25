#!/usr/bin/env bash
#
# 文件功能：启动统一审核网关服务。从 backend/gateway/app/main.py
# 加载 FastAPI 应用，通过 Uvicorn 在配置的地址和端口上启动。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"

exec "${PYTHON_BIN:-python}" -m uvicorn app.main:app \
  --app-dir backend/gateway \
  --host "${GATEWAY_HOST:-127.0.0.1}" \
  --port "${GATEWAY_PORT:-8080}"
