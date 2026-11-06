# v23: v21'den devam, genis taban + hedefli karisim (unutmayi azaltma denemesi).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v23.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v23 from v21 ep2 on mixed data (lr 3e-5, 4 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v23_mix.npz `
  --init data/mazlum/v21_epochs/epoch_002.npz --out data/mazlum/model_v23_mix.npz `
  --epochs 4 --lr 3e-5 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v23_epochs *>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "screen v23 checkpoints vs dragapult (200 each)"
python -u tools/screen_checkpoints.py --dir data/mazlum/v23_epochs `
  --opponent "$root\bots\dragapult" --games 200 --seed 777 `
  --out results/v23_screen.json *>&1 | Tee-Object -FilePath $log -Append

Step "done"
