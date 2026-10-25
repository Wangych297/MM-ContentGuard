#!/usr/bin/env bash
#
# 文件功能：下载视频审核服务运行时资产。从 GitHub Release 下载
# PCLMM 跨注意力模型、GoogleViT、BERT、FER-VT、ResNet34 和
# Whisper large-v3 等预训练权重，下载后执行视频服务的安装脚本。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_DIR="$ROOT_DIR/services/video"
VERSION="${VERSION:-v0.1.0}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/linsh63/multimodal-risk-chat/releases/download/$VERSION}"
ASSET_DIR="${ASSET_DIR:-$VIDEO_DIR/outputs/release_assets}"

assets=(
  "runtime-assets-$VERSION.sha256"
  "video-pclmm-runtime-core-$VERSION.tar.gz"
  "whisper-large-v3.pt.part-aa"
  "whisper-large-v3.pt.part-ab"
  "whisper-large-v3.pt.sha256"
)

mkdir -p "$ASSET_DIR"
cd "$ASSET_DIR"

for asset in "${assets[@]}"; do
  if [[ -f "$asset" ]]; then
    printf "[skip] %s already exists\n" "$asset"
  else
    printf "[download] %s\n" "$asset"
    curl --fail --location --remote-name "$RELEASE_DOWNLOAD_BASE_URL/$asset"
  fi
done

sha256sum -c "runtime-assets-$VERSION.sha256"

cd "$VIDEO_DIR"
VERSION="$VERSION" bash scripts/install_runtime_assets.sh

cd "$ROOT_DIR"

video_paths=(
  "services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth"
  "services/video/pretrained/googlevit-base-patch16-224-in21k/model.safetensors"
  "services/video/pretrained/bert_chinese/model.safetensors"
  "services/video/pretrained/FER-VT/model/FERVT.py"
  "services/video/pretrained/FER-VT/model/__init__.py"
  "services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth"
  "services/video/outputs/cache/whisper/large-v3.pt"
)

status=0
for path in "${video_paths[@]}"; do
  if [[ -e "$path" ]]; then
    printf "[ok]   %s\n" "$path"
  else
    printf "[miss] %s\n" "$path"
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  printf "\nVideo runtime assets are ready.\n"
  printf "Run bash scripts/check_runtime_assets.sh after text/image assets are ready.\n"
else
  printf "\nSome video runtime assets are missing.\n"
fi

exit "$status"
