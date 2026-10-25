# ============================================================
# 文件功能：查看图片分类模型训练进度。显示训练日志和指标变化。
# ============================================================
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Root 'checkpoints\image_model_dinov3_large'
$Progress = Join-Path $OutputDir 'progress.json'
$StdoutLog = Join-Path $OutputDir 'train_stdout.log'
$StderrLog = Join-Path $OutputDir 'train_stderr.log'
$PidFile = Join-Path $OutputDir 'train.pid'

if (Test-Path -Path $Progress) {
  Write-Output '== progress.json =='
  Get-Content -Path $Progress -Raw
} else {
  Write-Output 'No progress.json yet.'
}

if (Test-Path -Path $PidFile) {
  $pidValue = Get-Content -Path $PidFile -Raw
  Write-Output '== process =='
  Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, CPU, StartTime, HasExited |
    Format-List
}

if (Test-Path -Path $StdoutLog) {
  Write-Output '== train_stdout.log tail =='
  Get-Content -Path $StdoutLog -Tail 30
}

if (Test-Path -Path $StderrLog) {
  Write-Output '== train_stderr.log tail =='
  Get-Content -Path $StderrLog -Tail 30
}

Write-Output '== gpu =='
nvidia-smi
