# v31: big-arch Slowking fine-tuned on the full dataset (both sides, 168k rows).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v31.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v31 from v27 ep25 on full Slowking data (6 epochs, lr 2e-4)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v22_slowking.npz `
  --arch big --init data/mazlum/v27_epochs/epoch_025.npz --out data/mazlum/model_v31_big_full.npz `
  --epochs 6 --lr 2e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v31_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(3, 6)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v31_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v31_epochs\$file"
    Step "v31 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v31_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
