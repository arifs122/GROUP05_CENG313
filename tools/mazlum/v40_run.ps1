# v40: Dragapult specialist (500k rows, Dragapult-win episodes, feature v2).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v40.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "3"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v40 Dragapult big arch, 500k rows (25 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v40_dragapult_win.npz `
  --arch big --out data/mazlum/model_v40_dragapult.npz `
  --epochs 25 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v40_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(15, 20, 25)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v40_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v40_epochs\$file"
    Step "v40 $file vs dragapult bot (mirror, 300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_dragapult --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v40_ep${ep}_mirror_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}

$env:MAZLUM_MODEL = "$root\data\mazlum\v40_epochs\epoch_025.npz"
Step "v40 ep25 vs banette bot (300 games, seed 777)"
python -u tools/arena.py --bot0 bots/mazlum_dragapult --bot1 bots/banette --games 300 --seed 777 `
  --out "results/v40_ep25_banette_300.json" *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
