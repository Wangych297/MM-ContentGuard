#!/usr/bin/env bash
#
# 文件功能：解压 PCLMM 数据集。将下载的 PCLMM 数据集压缩包
# 解压到指定的数据目录，供训练和评测使用。
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_DIR="${ZIP_DIR:-$PROJECT_ROOT/data/raw/zips}"
VIDEO_DIR="${VIDEO_DIR:-$PROJECT_ROOT/data/raw/videos}"

FILES=(
  "children.zip"
  "disabled.zip"
  "elderly.zip"
  "low_income.zip"
  "single_parent.zip"
  "women.zip"
)

if [[ -n "${DATASET_FILES:-}" ]]; then
  read -r -a FILES <<< "$DATASET_FILES"
fi

mkdir -p "$VIDEO_DIR"

echo "Zip dir: $ZIP_DIR"
echo "Video dir: $VIDEO_DIR"
echo

for filename in "${FILES[@]}"; do
  zip_path="$ZIP_DIR/$filename"
  if [[ ! -f "$zip_path" ]]; then
    echo "Missing zip, skip: $zip_path"
    continue
  fi

  echo "Testing zip: $zip_path"
  unzip -tq "$zip_path"

  echo "Unpacking: $zip_path"
  unzip -n "$zip_path" -d "$VIDEO_DIR"
done

echo
echo "Unpack complete."
find "$VIDEO_DIR" -maxdepth 2 -type f -name '*.mp4' | wc -l | awk '{print "Total mp4 files:", $1}'
