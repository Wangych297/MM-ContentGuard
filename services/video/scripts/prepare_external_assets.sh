#!/usr/bin/env bash
#
# 文件功能：准备外部资产。下载和配置 PCLMM、FER-VT 等第三方
# 预训练模型所需的权重文件和依赖项。
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/2] Preparing pretrained models..."
bash "$PROJECT_ROOT/scripts/prepare_pretrained_models.sh"

echo
echo "[2/2] Preparing PCLMM dataset..."
bash "$PROJECT_ROOT/scripts/prepare_pclmm_dataset.sh"

echo
echo "All external assets are prepared."
