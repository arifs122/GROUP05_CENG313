# v30: continue generalist->specialist beyond 10 epochs (val still improving).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v30.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v30 from v29 ep10 (20 more epochs, lr 5e-5)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v26_slowking_only.npz `
  --init data/mazlum/v29_epochs/epoch_010.npz --out data/mazlum/model_v30_slowking_ft.npz `
  --epochs 20 --lr 5e-5 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v30_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(5, 10, 15, 20)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v30_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v30_epochs\$file"
    Step "v30 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v30_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
