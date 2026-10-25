# ============================================================
# 文件功能：训练图片风险分类模型。运行 model.train 和 model.evaluate。
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m model.train --config configs/image_classifier_dinov3_large.yaml --resume
python -m model.evaluate --config configs/image_classifier_dinov3_large.yaml




