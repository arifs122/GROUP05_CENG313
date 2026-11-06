# v35: squeeze v34 ep25 with a short low-lr fine-tune on the same 181k rows.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v35.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "3"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v35 from v34 ep25 (8 epochs, lr 3e-4)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v34_v2_slowking.npz `
  --arch big --init data/mazlum/v34_epochs/epoch_025.npz --out data/mazlum/model_v35_v2.npz `
  --epochs 8 --lr 3e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v35_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(4, 8)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v35_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v35_epochs\$file"
    Step "v35 $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v35_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
