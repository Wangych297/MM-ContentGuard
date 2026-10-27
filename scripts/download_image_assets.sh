#!/usr/bin/env bash
#
# 文件功能：下载图片审核服务运行时资产。从 GitHub Release 下载
# 模型检查点、DINOv3 权重、HuggingFace 缓存和 EasyOCR 模型，
# 支持分卷压缩包的下载与合并解压，带 SHA-256 完整性校验。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/Wangych297/MM-ContentGuard/releases/download/$VERSION}"
ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/release_assets/image}"

assets=(
  "image-runtime-assets-$VERSION.sha256"
  "image-runtime-parts-$VERSION.txt"
  "image-runtime-core-$VERSION.tar.gz.sha256"
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

parts_list="image-runtime-parts-$VERSION.txt"
while IFS= read -r part; do
  [[ -z "$part" ]] && continue
  if [[ -f "$part" ]]; then
    printf "[skip] %s already exists\n" "$part"
  else
    printf "[download] %s\n" "$part"
    curl --fail --location --remote-name "$RELEASE_DOWNLOAD_BASE_URL/$part"
  fi
done < "$parts_list"

sha256sum -c "image-runtime-assets-$VERSION.sha256"

archive_sha_file="image-runtime-core-$VERSION.tar.gz.sha256"
expected_sha="$(awk '{print $1}' "$archive_sha_file")"
actual_sha="$(while IFS= read -r part; do [[ -z "$part" ]] || cat "$part"; done < "$parts_list" | sha256sum | awk '{print $1}')"

if [[ "$actual_sha" != "$expected_sha" ]]; then
  printf "Reconstructed image archive checksum mismatch.\n" >&2
  printf "expected: %s\nactual:   %s\n" "$expected_sha" "$actual_sha" >&2
  exit 1
fi

cd "$ROOT_DIR"
(
  cd "$ASSET_DIR"
  while IFS= read -r part; do [[ -z "$part" ]] || cat "$part"; done < "$parts_list"
) | tar -xzf - -C "$ROOT_DIR"

required_paths=(
  "services/image/checkpoints/image_model_dinov3_large/best.pt"
  "services/image/checkpoints/semantic_model_focal_binary/best.pt"
  "services/image/.cache/manual_weights/dinov3_large/model.safetensors"
  "services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14"
  "services/image/.EasyOCR/model/craft_mlt_25k.pth"
  "services/image/.EasyOCR/model/zh_sim_g2.pth"
)

status=0
for path in "${required_paths[@]}"; do
  if [[ -e "$path" ]]; then
    printf "[ok]   %s\n" "$path"
  else
    printf "[miss] %s\n" "$path"
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  printf "\nImage runtime assets are ready.\n"
else
  printf "\nSome image runtime assets are missing.\n"
fi

exit "$status"
