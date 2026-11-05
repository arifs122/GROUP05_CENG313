# v28: win-weighted Slowking (Dragapult wins duplicated 3x), fine-tune from v26.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v28.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v28 from v26 ep30, Dragapult-win rows 3x (6 epochs, lr 5e-5)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v28_winweighted.npz `
  --init data/mazlum/v26_epochs/epoch_030.npz --out data/mazlum/model_v28_winweighted.npz `
  --epochs 6 --lr 5e-5 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v28_epochs *>&1 | Tee-Object -FilePath $log -Append

$env:MAZLUM_MODEL = "$root\data\mazlum\v28_epochs\epoch_006.npz"
Step "v28 ep6 vs dragapult (300 games, seed 777)"
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
  --out results/v28_ep6_dragapult_300.json *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue

Step "done"
