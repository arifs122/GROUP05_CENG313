# v29: generalist -> Slowking specialist (writeup recipe), from multi-deck BC.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v29.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v29 from multi-deck generalist on Slowking-side data (10 epochs, lr 1e-4)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v26_slowking_only.npz `
  --init data/mazlum/model_v14b_multi.npz --out data/mazlum/model_v29_slowking_ft.npz `
  --epochs 10 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v29_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(5, 10)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v29_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v29_epochs\$file"
    Step "v29 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v29_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
