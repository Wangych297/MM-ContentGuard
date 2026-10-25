#!/usr/bin/env bash
#
# 文件功能：检查所有服务的健康状态。依次向文本、图片、视频、网关、
# 前端五个服务发送健康检查请求，报告每个服务的通过/失败状态。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

TEXT_API_URL="${TEXT_API_URL:-http://127.0.0.1:8000}"
IMAGE_API_URL="${IMAGE_API_URL:-http://127.0.0.1:8010}"
VIDEO_API_URL="${VIDEO_API_URL:-http://127.0.0.1:8020}"
GATEWAY_URL="http://${GATEWAY_HOST:-127.0.0.1}:${GATEWAY_PORT:-8080}"
FRONTEND_URL="http://${FRONTEND_HOST:-127.0.0.1}:${FRONTEND_PORT:-5173}"

check_url() {
  local name="$1"
  local url="$2"
  if curl --noproxy "*" -fsS --max-time 5 "$url" >/dev/null; then
    printf "[ok]   %s %s\n" "$name" "$url"
  else
    printf "[fail] %s %s\n" "$name" "$url"
    return 1
  fi
}

status=0
check_url "text" "$TEXT_API_URL/health" || status=1
check_url "image" "$IMAGE_API_URL/health" || status=1
check_url "video" "$VIDEO_API_URL/health" || status=1
check_url "gateway" "$GATEWAY_URL/api/health" || status=1
check_url "frontend" "$FRONTEND_URL/" || status=1

exit "$status"
