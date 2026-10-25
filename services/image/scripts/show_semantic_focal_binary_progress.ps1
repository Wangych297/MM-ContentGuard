# ============================================================
# 文件功能：查看语义 Focal Binary 模型训练进度。
# ============================================================
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Root 'checkpoints\semantic_model_focal_binary'
$ProgressPath = Join-Path $OutputDir 'progress.json'

if (Test-Path $ProgressPath) {
  Get-Content $ProgressPath -Raw
} else {
  'NO_PROGRESS'
}
