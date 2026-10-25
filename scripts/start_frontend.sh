#!/usr/bin/env bash
#
# 文件功能：启动前端服务。FRONTEND_MODE=dev 时启动 Vite 开发服务器，
# 否则执行 npm run build 构建后通过 serve_frontend_static.py 提供
# 静态文件服务。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/frontend/web"

if [[ ! -d node_modules ]]; then
  npm install
fi

if [[ "${FRONTEND_MODE:-static}" == "dev" ]]; then
  exec ./node_modules/.bin/vite --host "${FRONTEND_HOST:-127.0.0.1}" --port "${FRONTEND_PORT:-5173}" --strictPort
fi

if [[ "${FRONTEND_SKIP_BUILD:-0}" != "1" ]]; then
  npm run build
fi

cd "$ROOT_DIR"
exec "${PYTHON_BIN:-python}" "$ROOT_DIR/scripts/serve_frontend_static.py"
