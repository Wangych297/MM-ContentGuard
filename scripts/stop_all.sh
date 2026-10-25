#!/usr/bin/env bash
#
# 文件功能：停止所有运行中的服务。按前端→网关→视频→图片→文本的
# 逆序依次停止，优先使用 PID 文件，PID 文件不存在时通过 ss 命令
# 检测端口占用并 kill。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/logs/pids"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

pid_from_port() {
  local port="$1"
  ss -ltnp "sport = :$port" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]*\).*/\1/p' \
    | head -1
}

stop_service() {
  local name="$1"
  local port="$2"
  local pid_file="$PID_DIR/$name.pid"
  local pid=""

  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
  fi

  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    pid="$(pid_from_port "$port")"
  fi

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    printf "[stop] %s pid=%s\n" "$name" "$pid"
    kill -- "-$pid" 2>/dev/null || kill "$pid"
  else
    printf "[skip] %s is not running\n" "$name"
  fi
  rm -f "$pid_file"
}

stop_service frontend "${FRONTEND_PORT:-5173}"
stop_service gateway "${GATEWAY_PORT:-8080}"
stop_service video "${VIDEO_PORT:-8020}"
stop_service image "${IMAGE_PORT:-8010}"
stop_service text "${TEXT_PORT:-8000}"
