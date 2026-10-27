#!/usr/bin/env bash
#
# 文件功能：下载 Demo 演示素材。从 GitHub Release 下载示例视频等
# 演示媒体文件，校验后解压到 demo/ 目录，用于测试视频审核功能。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/Wangych297/MM-ContentGuard/releases/download/$VERSION}"
ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/release_assets/demo}"

assets=(
  "demo-assets-$VERSION.sha256"
  "demo-assets-$VERSION.tar.gz"
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

sha256sum -c "demo-assets-$VERSION.sha256"

cd "$ROOT_DIR"
tar -xzf "$ASSET_DIR/demo-assets-$VERSION.tar.gz" -C "$ROOT_DIR"

printf "\nDemo assets are ready under demo/.\n"

