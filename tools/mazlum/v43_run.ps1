# v43: Banette, all episodes (both sides), feature v2, big arch.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v43.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "5"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v43 big arch on all Banette data (700k rows, 12 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v43_banette.npz `
  --arch big --out data/mazlum/model_v43_banette.npz `
  --epochs 12 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v43_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(6, 12)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v43_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v43_epochs\$file"
    foreach ($opp in @("dragapult", "public_alakazam", "public_meta")) {
        Step "v43 $file vs $opp (100 games, seed 777)"
        python -u tools/arena.py --bot0 bots/mazlum --bot1 "bots/$opp" --games 100 --seed 777 `
          --out "results/v43_ep${ep}_${opp}_100.json" *>&1 |
          Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
    }
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
