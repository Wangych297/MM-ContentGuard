#!/usr/bin/env bash
#
# 文件功能：安装运行时资产。将 Release 资产解压到视频服务的
# 对应目录（模型检查点、预训练权重、配置文件等）。
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$PROJECT_ROOT/outputs/release_assets}"
VERSION="${VERSION:-v0.1.0}"
CORE_ARCHIVE="${CORE_ARCHIVE:-$ASSET_DIR/video-pclmm-runtime-core-${VERSION}.tar.gz}"
WHISPER_DIR="${WHISPER_DIR:-$PROJECT_ROOT/outputs/cache/whisper}"

if [[ ! -f "$CORE_ARCHIVE" ]]; then
  echo "Core runtime archive not found: $CORE_ARCHIVE" >&2
  echo "Download the GitHub Release assets first, or set CORE_ARCHIVE=/path/to/archive." >&2
  exit 1
fi

echo "Extracting core runtime assets from: $CORE_ARCHIVE"
tar -xzf "$CORE_ARCHIVE" -C "$PROJECT_ROOT"

# The integrated monorepo also contains the image service's top-level `model`
# package. Make FER-VT's model directory an explicit package so imports resolve
# to the video service copy first.
touch "$PROJECT_ROOT/pretrained/FER-VT/model/__init__.py"

shopt -s nullglob
whisper_parts=("$ASSET_DIR"/whisper-large-v3.pt.part-*)
if (( ${#whisper_parts[@]} > 0 )); then
  mkdir -p "$WHISPER_DIR"
  whisper_download_name="$WHISPER_DIR/whisper-large-v3.pt"
  whisper_runtime_name="$WHISPER_DIR/large-v3.pt"
  echo "Reconstructing Whisper model at: $whisper_download_name"
  cat "${whisper_parts[@]}" > "$whisper_download_name"

  if [[ -f "$ASSET_DIR/whisper-large-v3.pt.sha256" ]]; then
    (
      cd "$WHISPER_DIR"
      sha256sum -c "$ASSET_DIR/whisper-large-v3.pt.sha256"
    )
  fi
  mv "$whisper_download_name" "$whisper_runtime_name"
else
  echo "No Whisper split parts found under $ASSET_DIR; skipping Whisper install."
fi

echo "Runtime assets installed."
