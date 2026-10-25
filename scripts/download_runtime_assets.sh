#!/usr/bin/env bash
#
# 文件功能：一键下载所有运行时资产。依次调用 download_text_assets.sh、
# download_image_assets.sh、download_video_assets.sh，下载完成后
# 运行 check_runtime_assets.sh 验证完整性。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-v0.1.0}"
export VERSION

bash scripts/download_text_assets.sh
bash scripts/download_image_assets.sh
bash scripts/download_video_assets.sh

bash scripts/check_runtime_assets.sh

printf "\nOptional datasets and demo media are not required for inference.\n"
printf "Download them only if needed:\n"
printf "  VERSION=%s bash scripts/download_dataset_assets.sh\n" "$VERSION"
printf "  VERSION=%s bash scripts/download_demo_assets.sh\n" "$VERSION"
