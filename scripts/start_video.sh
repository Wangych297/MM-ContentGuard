#!/usr/bin/env bash
#
# 文件功能：启动视频审核服务。使用 PCLMM 多模态模型，配置
# VIT/Whisper/BERT/Face 子模型的 GPU 设备映射、特征提取模式和
# 审核阈值，通过 Uvicorn 启动 API 服务。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/services/video"

mkdir -p pretrained/FER-VT/model
touch pretrained/FER-VT/model/__init__.py

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PCLMM_API_FEATURE_BACKEND="${PCLMM_API_FEATURE_BACKEND:-resident}"
export PCLMM_API_CUDA_VISIBLE_DEVICES="${VIDEO_CUDA_VISIBLE_DEVICES:-6,7}"
export PCLMM_API_DEVICE="${PCLMM_API_DEVICE:-cuda:1}"
export PCLMM_API_VIT_DEVICE="${PCLMM_API_VIT_DEVICE:-cuda:0}"
export PCLMM_API_WHISPER_DEVICE="${PCLMM_API_WHISPER_DEVICE:-cuda:0}"
export PCLMM_API_BERT_DEVICE="${PCLMM_API_BERT_DEVICE:-cuda:0}"
export PCLMM_API_FACE_DEVICE="${PCLMM_API_FACE_DEVICE:-cuda:1}"
export PCLMM_API_EXTRACTION_MODE="${PCLMM_API_EXTRACTION_MODE:-parallel}"
export PCLMM_API_THRESHOLD="${VIDEO_THRESHOLD:-0.32}"
export PCLMM_API_VIT_BATCH_SIZE="${PCLMM_API_VIT_BATCH_SIZE:-16}"
export PCLMM_API_FACE_MAX_SAMPLES="${PCLMM_API_FACE_MAX_SAMPLES:-8}"
export WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-$PWD/outputs/cache/whisper}"
export TORCH_HOME="${TORCH_HOME:-$PWD/outputs/cache/torch}"
export HOST="${VIDEO_HOST:-127.0.0.1}"
export PORT="${VIDEO_PORT:-8020}"

exec scripts/run_api.sh
