# ============================================================
# 文件功能：后台启动语义 Focal Binary 模型训练。
# ============================================================
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Root 'checkpoints\semantic_model_focal_binary'
$CacheDir = Join-Path $Root '.cache'
$ConfigPath = 'configs/semantic_meme_focal_binary.yaml'
$InitCheckpoint = Join-Path $Root 'checkpoints\semantic_model\best.pt'
$LastCheckpoint = Join-Path $OutputDir 'last.pt'

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$StdoutLog = Join-Path $OutputDir 'train_stdout.log'
$StderrLog = Join-Path $OutputDir 'train_stderr.log'
$PidFile = Join-Path $OutputDir 'train.pid'

$PythonExe = (Get-Command python).Source
[Environment]::SetEnvironmentVariable('TORCH_HOME', (Join-Path $CacheDir 'torch'), 'Process')
[Environment]::SetEnvironmentVariable('PYTHONUNBUFFERED', '1', 'Process')

if (Test-Path $LastCheckpoint) {
  $TrainArgs = "-m semantic_model.train --config $ConfigPath --resume"
} else {
  if (-not (Test-Path $InitCheckpoint)) {
    throw "Init checkpoint not found: $InitCheckpoint"
  }
  $TrainArgs = "-m semantic_model.train --config $ConfigPath --init-checkpoint `"$InitCheckpoint`""
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $env:COMSPEC
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$psi.Arguments = "/d /c `"`"$PythonExe`" $TrainArgs 1> `"$StdoutLog`" 2> `"$StderrLog`"`""

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$null = $proc.Start()

$proc.Id | Set-Content -Path $PidFile -Encoding ASCII

[pscustomobject]@{
  pid = $proc.Id
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  progress = (Join-Path $OutputDir 'progress.json')
  last_checkpoint = $LastCheckpoint
  best_checkpoint = (Join-Path $OutputDir 'best.pt')
  config = (Join-Path $Root $ConfigPath)
  mode = if (Test-Path $LastCheckpoint) { 'resume' } else { 'init_checkpoint' }
} | ConvertTo-Json -Depth 3
