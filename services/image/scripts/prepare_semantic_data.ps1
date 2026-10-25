# ============================================================
# 文件功能：准备语义 Meme 模型数据。对语义标注数据进行划分和标准化。
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m semantic_model.prepare_data --config configs/semantic_meme.yaml
