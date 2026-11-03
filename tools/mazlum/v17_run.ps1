# v17: massive targeted dataset (2k+ pairs), then fine-tune from v13.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v17_train.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "build v17 massive dataset (Banette vs Dragapult)"
python -u tools/mazlum/build_targeted_dataset.py --capacity 900000 `
  --out data/mazlum/dataset_v17_dragapult_massive.npz 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "build FAILED"; exit 1 }

Step "fine-tune v13 -> v17 (lr 1e-4, 15 epochs, save-each)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v17_dragapult_massive.npz `
  --init data/mazlum/model_banette_v13.npz --out data/mazlum/model_v17_massive.npz `
  --epochs 15 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v17_epochs 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "done"
