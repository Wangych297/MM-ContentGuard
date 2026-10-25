# ============================================================
# 文件功能：训练语义 Meme 分类模型。运行训练和测试集评测。
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m semantic_model.train --config configs/semantic_meme.yaml --resume
python -m semantic_model.evaluate --config configs/semantic_meme.yaml --split test
