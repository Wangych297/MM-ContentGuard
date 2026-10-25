# ============================================================
# 文件功能：准备图片训练数据。构建数据清单并执行分层划分。
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m data.build_manifest --config configs/image_classifier_dinov3_large.yaml
python -m data.split --config configs/image_classifier_dinov3_large.yaml



