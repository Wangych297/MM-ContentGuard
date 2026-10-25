#!/usr/bin/env bash
#
# 文件功能：打包 Release 资产。将文本、图片、视频的模型权重和
# 运行缓存打包为版本化的 .tar.gz 压缩包，生成 SHA-256 清单文件，
# 支持可选打包数据集和 Demo 素材。生成的资产位于 release_assets/ 目录。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/release_assets}"
PART_SIZE="${PART_SIZE:-1900M}"

PACKAGE_TEXT="${PACKAGE_TEXT:-true}"
PACKAGE_IMAGE="${PACKAGE_IMAGE:-true}"
PACKAGE_VIDEO="${PACKAGE_VIDEO:-true}"
PACKAGE_DEMO="${PACKAGE_DEMO:-false}"
PACKAGE_DATASETS="${PACKAGE_DATASETS:-false}"

mkdir -p "$OUT_DIR"

check_required_paths() {
  local missing=0
  local path
  for path in "$@"; do
    if [[ ! -e "$ROOT_DIR/$path" ]]; then
      printf "[miss] %s\n" "$path" >&2
      missing=1
    else
      printf "[ok]   %s\n" "$path"
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
}

write_manifest() {
  local manifest="$1"
  shift
  (
    cd "$OUT_DIR"
    sha256sum "$@" > "$manifest"
  )
}

package_text_assets() {
  local archive="text-runtime-core-$VERSION.tar.gz"
  local manifest="text-runtime-assets-$VERSION.sha256"
  local paths=(
    "services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug"
    "services/text/runs/social_intent_binary_v1_from_general"
    "services/text/runs/reason_category_v1_macbert"
    "services/text/models/hfl/chinese-macbert-base"
    "services/text/data/raw/ast"
  )

  printf "\n== Text runtime assets ==\n"
  check_required_paths "${paths[@]}"
  rm -f "$OUT_DIR/$archive" "$OUT_DIR/$manifest"

  (
    cd "$ROOT_DIR"
    tar --dereference \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf "$OUT_DIR/$archive" \
      "${paths[@]}"
  )

  write_manifest "$manifest" "$archive"
  printf "[ready] %s\n" "$OUT_DIR/$archive"
  printf "[ready] %s\n" "$OUT_DIR/$manifest"
}

package_image_assets() {
  local archive="image-runtime-core-$VERSION.tar.gz"
  local archive_sha="$archive.sha256"
  local parts_list="image-runtime-parts-$VERSION.txt"
  local manifest="image-runtime-assets-$VERSION.sha256"
  local paths=(
    "services/image/checkpoints/image_model_dinov3_large/best.pt"
    "services/image/checkpoints/semantic_model_focal_binary/best.pt"
    "services/image/.cache/manual_weights/dinov3_large/model.safetensors"
    "services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14"
    "services/image/.EasyOCR/model"
  )

  printf "\n== Image runtime assets ==\n"
  check_required_paths "${paths[@]}"
  rm -f \
    "$OUT_DIR/$archive" \
    "$OUT_DIR/$archive".part-* \
    "$OUT_DIR/$archive_sha" \
    "$OUT_DIR/$parts_list" \
    "$OUT_DIR/$manifest"

  printf "[pack] streaming %s into %s parts\n" "$archive" "$PART_SIZE"
  (
    cd "$ROOT_DIR"
    tar --dereference \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf - \
      "${paths[@]}" \
      | tee >(sha256sum | awk -v name="$archive" '{print $1 "  " name}' > "$OUT_DIR/$archive_sha") \
      | split -b "$PART_SIZE" - "$OUT_DIR/$archive.part-"
  )

  while [[ ! -s "$OUT_DIR/$archive_sha" ]]; do
    sleep 0.2
  done

  (
    cd "$OUT_DIR"
    ls "$archive".part-* | sort > "$parts_list"
    sha256sum "$parts_list" "$archive_sha" "$archive".part-* > "$manifest"
  )

  printf "[ready] %s\n" "$OUT_DIR/$parts_list"
  printf "[ready] %s\n" "$OUT_DIR/$archive_sha"
  printf "[ready] %s\n" "$OUT_DIR/$manifest"
}

package_video_assets() {
  local whisper_file="$ROOT_DIR/services/video/outputs/cache/whisper/large-v3.pt"

  printf "\n== Video runtime assets ==\n"
  if [[ ! -f "$whisper_file" ]]; then
    printf "[miss] %s\n" "services/video/outputs/cache/whisper/large-v3.pt" >&2
    return 1
  fi

  VERSION="$VERSION" \
  OUT_DIR="$OUT_DIR" \
  PART_SIZE="$PART_SIZE" \
  WHISPER_FILE="$whisper_file" \
  bash "$ROOT_DIR/services/video/scripts/package_runtime_assets.sh"
}

package_demo_assets() {
  local archive="demo-assets-$VERSION.tar.gz"
  local manifest="demo-assets-$VERSION.sha256"

  printf "\n== Demo assets ==\n"
  if [[ ! -d "$ROOT_DIR/demo" ]]; then
    printf "[skip] demo/ does not exist\n"
    return 0
  fi

  rm -f "$OUT_DIR/$archive" "$OUT_DIR/$manifest"
  (
    cd "$ROOT_DIR"
    tar --dereference \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf "$OUT_DIR/$archive" \
      demo
  )

  write_manifest "$manifest" "$archive"
  printf "[ready] %s\n" "$OUT_DIR/$archive"
  printf "[ready] %s\n" "$OUT_DIR/$manifest"
}

package_dataset_assets() {
  local archive="dataset-assets-$VERSION.tar.gz"
  local manifest="dataset-assets-$VERSION.sha256"
  local paths=()

  printf "\n== Optional dataset assets ==\n"

  if [[ -d "$ROOT_DIR/storage/datasets" ]]; then
    paths+=("storage/datasets")
  fi

  if [[ -d "$ROOT_DIR/services/text/data/processed" ]]; then
    paths+=("services/text/data/processed")
  fi

  if [[ -d "$ROOT_DIR/services/image/data/processed" ]]; then
    paths+=("services/image/data/processed")
  fi

  if [[ -d "$ROOT_DIR/services/video/data/processed" ]]; then
    paths+=("services/video/data/processed")
  fi

  if (( ${#paths[@]} == 0 )); then
    printf "[skip] no optional dataset directories found\n"
    return 0
  fi

  rm -f "$OUT_DIR/$archive" "$OUT_DIR/$manifest"
  (
    cd "$ROOT_DIR"
    tar --dereference \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf "$OUT_DIR/$archive" \
      "${paths[@]}"
  )

  write_manifest "$manifest" "$archive"
  printf "[ready] %s\n" "$OUT_DIR/$archive"
  printf "[ready] %s\n" "$OUT_DIR/$manifest"
}

cd "$ROOT_DIR"

if [[ "$PACKAGE_TEXT" == "true" ]]; then
  package_text_assets
fi

if [[ "$PACKAGE_IMAGE" == "true" ]]; then
  package_image_assets
fi

if [[ "$PACKAGE_VIDEO" == "true" ]]; then
  package_video_assets
fi

if [[ "$PACKAGE_DEMO" == "true" ]]; then
  package_demo_assets
fi

if [[ "$PACKAGE_DATASETS" == "true" ]]; then
  package_dataset_assets
fi

printf "\nRelease assets are under:\n  %s\n\n" "$OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -type f -printf "%f\n" | sort
