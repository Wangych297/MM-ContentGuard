#!/usr/bin/env bash
#
# 文件功能：一键后台启动所有服务（文本、图片、视频、网关、前端）。
# 通过 setsid 在后台启动各服务，管理 PID 文件，并轮询健康检查端点
# 等待所有服务就绪。
#
# 核心步骤：
#   1. 依次启动五个子服务脚本
#   2. 轮询 check_services.sh 直到全部通过或超时
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$LOG_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

is_port_listening() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

wait_url() {
  local name="$1"
  local url="$2"
  local timeout="$3"
  local elapsed=0

  printf "[wait] %s %s" "$name" "$url"
  while (( elapsed < timeout )); do
    if curl --noproxy "*" -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      printf "\r[ok]   %s %s\n" "$name" "$url"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    printf "."
  done

  printf "\r[fail] %s %s\n" "$name" "$url"
  return 1
}

start_service() {
  local name="$1"
  local script="$2"
  local port="$3"
  local pid_file="$PID_DIR/$name.pid"
  local log_file="$LOG_DIR/$name.log"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf "[skip] %s already running, pid=%s\n" "$name" "$(cat "$pid_file")"
    return
  fi

  if is_port_listening "$port"; then
    printf "[skip] %s port %s is already listening\n" "$name" "$port"
    return
  fi

  printf "[start] %s -> %s\n" "$name" "$log_file"
  (
    cd "$ROOT_DIR"
    setsid bash "$script" >"$log_file" 2>&1 < /dev/null &
    echo "$!" >"$pid_file"
  )
}

start_service text "$ROOT_DIR/scripts/start_text.sh" "${TEXT_PORT:-8000}"
start_service image "$ROOT_DIR/scripts/start_image.sh" "${IMAGE_PORT:-8010}"
start_service video "$ROOT_DIR/scripts/start_video.sh" "${VIDEO_PORT:-8020}"
start_service gateway "$ROOT_DIR/scripts/start_gateway.sh" "${GATEWAY_PORT:-8080}"
start_service frontend "$ROOT_DIR/scripts/start_frontend.sh" "${FRONTEND_PORT:-5173}"

TEXT_API_URL="${TEXT_API_URL:-http://127.0.0.1:${TEXT_PORT:-8000}}"
IMAGE_API_URL="${IMAGE_API_URL:-http://127.0.0.1:${IMAGE_PORT:-8010}}"
VIDEO_API_URL="${VIDEO_API_URL:-http://127.0.0.1:${VIDEO_PORT:-8020}}"
GATEWAY_URL="http://${GATEWAY_HOST:-127.0.0.1}:${GATEWAY_PORT:-8080}"
FRONTEND_URL="http://${FRONTEND_HOST:-127.0.0.1}:${FRONTEND_PORT:-5173}"
START_ALL_WAIT_SECONDS="${START_ALL_WAIT_SECONDS:-180}"

printf "\nWaiting for services. Video may take longer on first load.\n"
status=0
wait_url text "$TEXT_API_URL/health" "$START_ALL_WAIT_SECONDS" || status=1
wait_url image "$IMAGE_API_URL/health" "$START_ALL_WAIT_SECONDS" || status=1
wait_url video "$VIDEO_API_URL/health" "$START_ALL_WAIT_SECONDS" || status=1
wait_url gateway "$GATEWAY_URL/api/health" "$START_ALL_WAIT_SECONDS" || status=1
wait_url frontend "$FRONTEND_URL/" 30 || status=1

printf "\nFrontend:\n"
printf "  %s/\n" "$FRONTEND_URL"

if (( status != 0 )); then
  printf "\nSome services are not ready. Check logs under:\n"
  printf "  %s\n" "$LOG_DIR"
  exit "$status"
fi

printf "\nAll services are ready.\n"
