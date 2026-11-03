# v17 resume: complete epochs 13-15 from epoch_012, then expose them in v17_epochs.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v17_resume.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === resume v17: epochs 13-15 from epoch_012 ===" | Tee-Object -FilePath $log -Append

python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v17_dragapult_massive.npz `
  --init data/mazlum/v17_epochs/epoch_012.npz --out data/mazlum/model_v17_massive.npz `
  --epochs 3 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v17_epochs_cont 2>&1 | Tee-Object -FilePath $log -Append

if ($LASTEXITCODE -ne 0) { "resume FAILED" | Tee-Object -FilePath $log -Append; exit 1 }

Copy-Item data\mazlum\v17_epochs_cont\epoch_001.npz data\mazlum\v17_epochs\epoch_013.npz -Force
Copy-Item data\mazlum\v17_epochs_cont\epoch_002.npz data\mazlum\v17_epochs\epoch_014.npz -Force
Copy-Item data\mazlum\v17_epochs_cont\epoch_003.npz data\mazlum\v17_epochs\epoch_015.npz -Force

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === done ===" | Tee-Object -FilePath $log -Append
