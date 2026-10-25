# ============================================================
# 文件功能：查看 Strong Image 分类模型训练进度。
# ============================================================
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Root 'checkpoints\image_model_strong'
$Progress = Join-Path $OutputDir 'progress.json'
$StdoutLog = Join-Path $OutputDir 'train_stdout.log'
$StderrLog = Join-Path $OutputDir 'train_stderr.log'
$PidFile = Join-Path $OutputDir 'train.pid'

if (Test-Path $Progress) {
  Get-Content $Progress -Raw
} else {
  Write-Host "No progress file yet: $Progress"
}

if (Test-Path $PidFile) {
  $TrainPid = [int](Get-Content $PidFile -Raw)
  Get-Process -Id $TrainPid -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,CPU,StartTime,Responding |
    Format-List
}

if (Test-Path $StdoutLog) {
  Write-Host "`n--- stdout tail ---"
  Get-Content $StdoutLog -Tail 30
}

if (Test-Path $StderrLog) {
  Write-Host "`n--- stderr tail ---"
  Get-Content $StderrLog -Tail 20
}

Write-Host "`n--- nvidia-smi ---"
nvidia-smi
