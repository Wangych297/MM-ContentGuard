#!/usr/bin/env bash
#
# 文件功能：下载可选数据集资产。从 GitHub Release 下载数据集压缩包，
# 通过 SHA-256 校验后解压到项目根目录。数据集非启动必需项，仅供
# 查看数据来源或复现实验使用。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/Wangych297/MM-ContentGuard/releases/download/$VERSION}"
ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/release_assets/datasets}"

assets=(
  "dataset-assets-$VERSION.sha256"
  "dataset-assets-$VERSION.tar.gz"
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

sha256sum -c "dataset-assets-$VERSION.sha256"

cd "$ROOT_DIR"
tar -xzf "$ASSET_DIR/dataset-assets-$VERSION.tar.gz" -C "$ROOT_DIR"

printf "\nOptional dataset assets are ready.\n"

