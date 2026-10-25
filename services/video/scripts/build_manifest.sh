#!/usr/bin/env bash
#
# 文件功能：构建视频数据清单。运行 build_manifest.py 扫描视频目录，
# 使用 ffprobe 提取视频元数据并生成 JSON 清单文件。
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ANNOTATION_FILE="${ANNOTATION_FILE:-$PROJECT_ROOT/data/annotations/Annotation_Subset.csv}"
VIDEO_ROOT="${VIDEO_ROOT:-$PROJECT_ROOT/data/raw/videos}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/data/processed/video_manifest.csv}"

"$PYTHON_BIN" "$PROJECT_ROOT/src/data/build_manifest.py" \
  --annotation-file "$ANNOTATION_FILE" \
  --video-root "$VIDEO_ROOT" \
  --output "$OUTPUT" \
  "$@"
