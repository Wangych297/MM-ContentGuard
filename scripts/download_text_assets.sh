#!/usr/bin/env bash
#
# 文件功能：下载文本审核服务运行时资产。从 GitHub Release 下载
# BERT 模型检查点、MacBERT 基础模型和 AST 数据文件，
# 通过 SHA-256 校验和验证完整性后解压。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/linsh63/multimodal-risk-chat/releases/download/$VERSION}"
ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/release_assets/text}"

assets=(
  "text-runtime-assets-$VERSION.sha256"
  "text-runtime-core-$VERSION.tar.gz"
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

sha256sum -c "text-runtime-assets-$VERSION.sha256"

cd "$ROOT_DIR"
tar -xzf "$ASSET_DIR/text-runtime-core-$VERSION.tar.gz" -C "$ROOT_DIR"

required_paths=(
  "services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug/pytorch_model.bin"
  "services/text/runs/social_intent_binary_v1_from_general/pytorch_model.bin"
  "services/text/runs/reason_category_v1_macbert/pytorch_model.bin"
  "services/text/models/hfl/chinese-macbert-base/pytorch_model.bin"
  "services/text/data/raw/ast/hanzi.txt"
  "services/text/data/raw/ast/similarity_matrix.pkl"
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
  printf "\nText runtime assets are ready.\n"
else
  printf "\nSome text runtime assets are missing.\n"
fi

exit "$status"

