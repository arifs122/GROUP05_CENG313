# v33: feature-schema v2 (47/44 dims), big arch, same rows as v27 for a clean A/B.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v33.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "3"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v33 big arch on feature-v2 data (25 epochs, lr 1e-3)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v33_v2_slowking.npz `
  --arch big --out data/mazlum/model_v33_v2.npz `
  --epochs 25 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v33_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(15, 20, 25)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v33_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v33_epochs\$file"
    Step "v33 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v33_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
