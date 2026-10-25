# ============================================================
# 文件功能：语义模型冒烟测试。在少量样本上快速验证模型推理流程。
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m semantic_model.smoke_batch --config configs/semantic_meme.yaml --batch-size 4
