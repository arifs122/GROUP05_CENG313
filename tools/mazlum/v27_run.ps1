# v27: big-architecture Slowking specialist (capacity test), from scratch on side-only rows.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v27.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "4"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v27 big arch, Slowking-side only (40 epochs, lr 1e-3)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v26_slowking_only.npz `
  --arch big --out data/mazlum/model_v27_big_slowking.npz `
  --epochs 40 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v27_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(10, 25, 40)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v27_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v27_epochs\$file"
    Step "v27 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v27_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
