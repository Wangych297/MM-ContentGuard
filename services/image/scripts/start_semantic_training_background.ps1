# ============================================================
# 文件功能：后台启动语义 Meme 模型训练。
# ============================================================
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Root 'checkpoints\semantic_model'
$CacheDir = Join-Path $Root '.cache'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$StdoutLog = Join-Path $OutputDir 'train_stdout.log'
$StderrLog = Join-Path $OutputDir 'train_stderr.log'
$PidFile = Join-Path $OutputDir 'train.pid'

$PythonExe = (Get-Command python).Source
[Environment]::SetEnvironmentVariable('TORCH_HOME', (Join-Path $CacheDir 'torch'), 'Process')
[Environment]::SetEnvironmentVariable('PYTHONUNBUFFERED', '1', 'Process')

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $env:COMSPEC
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$psi.Arguments = "/d /c `"`"$PythonExe`" -m semantic_model.train --config configs/semantic_meme.yaml --resume 1> `"$StdoutLog`" 2> `"$StderrLog`"`""

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$null = $proc.Start()

$proc.Id | Set-Content -Path $PidFile -Encoding ASCII

[pscustomobject]@{
  pid = $proc.Id
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  progress = (Join-Path $OutputDir 'progress.json')
  last_checkpoint = (Join-Path $OutputDir 'last.pt')
  best_checkpoint = (Join-Path $OutputDir 'best.pt')
} | ConvertTo-Json -Depth 3
