#!/usr/bin/env bash
#
# 文件功能：启动图片审核服务。配置 GPU 设备、HuggingFace 缓存、
# EasyOCR 模型路径和文本 API 转发地址，通过 Uvicorn 启动 API 服务。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/services/image"

export CUDA_VISIBLE_DEVICES="${IMAGE_CUDA_VISIBLE_DEVICES:-5}"
export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export EASYOCR_MODULE_PATH="${EASYOCR_MODULE_PATH:-$PWD/.EasyOCR}"
export RISK_IMAGE_OCR_GPU="${RISK_IMAGE_OCR_GPU:-true}"
TEXT_API_BASE="${TEXT_API_URL:-http://127.0.0.1:8000}"
export RISK_IMAGE_TEXT_API_URL="${RISK_IMAGE_TEXT_API_URL:-${TEXT_API_BASE%/}/text/check}"
export RISK_IMAGE_TEXT_API_TIMEOUT="${RISK_IMAGE_TEXT_API_TIMEOUT:-${TEXT_API_TIMEOUT:-15}}"

exec "${PYTHON_BIN:-python}" -m uvicorn api.app:app \
  --host "${IMAGE_HOST:-127.0.0.1}" \
  --port "${IMAGE_PORT:-8010}"
