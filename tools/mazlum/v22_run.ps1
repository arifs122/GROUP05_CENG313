# v22: Slowking specialist from scratch (37-dim, v9 arch).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v22.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === v22 Slowking BC cold start (25 epochs, lr 1e-3) ===" | Tee-Object -FilePath $log -Append

python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v22_slowking.npz `
  --out data/mazlum/model_v22_slowking.npz `
  --epochs 25 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v22_epochs *>&1 | Tee-Object -FilePath $log -Append

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === done ===" | Tee-Object -FilePath $log -Append
