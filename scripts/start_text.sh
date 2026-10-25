#!/usr/bin/env bash
#
# 文件功能：启动文本审核服务。使用 risk_text 后端，加载 BERT 模型
# 检查点，配置风险和可疑阈值，通过 Uvicorn 启动 API 服务。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

TEXT_API_BACKEND="${TEXT_API_BACKEND:-risk_text}"

if [[ "$TEXT_API_BACKEND" != "risk_text" ]]; then
  echo "Unsupported TEXT_API_BACKEND: $TEXT_API_BACKEND" >&2
  echo "Use TEXT_API_BACKEND=risk_text for the single-repo demo." >&2
  exit 1
fi

cd "$ROOT_DIR/services/text"

export CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="${PYTHONPATH:-src}"
export RISK_TEXT_HOST="${TEXT_HOST:-127.0.0.1}"
export RISK_TEXT_PORT="${TEXT_PORT:-8000}"
export RISK_TEXT_DEVICE="${TEXT_DEVICE:-cuda}"
export RISK_TEXT_GENERAL_CHECKPOINT="${RISK_TEXT_GENERAL_CHECKPOINT:-runs/social_binary_v3_macbert_binary_bs128_no_online_aug}"
export RISK_TEXT_SOCIAL_CHECKPOINT="${RISK_TEXT_SOCIAL_CHECKPOINT:-runs/social_intent_binary_v1_from_general}"
export RISK_TEXT_REASON_CHECKPOINT="${RISK_TEXT_REASON_CHECKPOINT:-runs/reason_category_v1_macbert}"
export RISK_TEXT_RISK_THRESHOLD="${RISK_TEXT_RISK_THRESHOLD:-${TEXT_RISK_THRESHOLD:-0.5}}"
export RISK_TEXT_SUSPICIOUS_THRESHOLD="${RISK_TEXT_SUSPICIOUS_THRESHOLD:-${TEXT_SUSPICIOUS_THRESHOLD:-0.5}}"
export RISK_TEXT_PARALLEL_BINARY="${RISK_TEXT_PARALLEL_BINARY:-true}"
export RISK_TEXT_PARALLEL_REASON="${RISK_TEXT_PARALLEL_REASON:-true}"

exec "${PYTHON_BIN:-python}" -m risk_text_detection.api.app
