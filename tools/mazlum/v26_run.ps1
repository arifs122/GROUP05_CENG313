# v26: Slowking-side-only BC (specialist policy), from scratch.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v26.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v26 from scratch, Slowking-side rows only (30 epochs, lr 1e-3)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v26_slowking_only.npz `
  --out data/mazlum/model_v26_slowking_only.npz `
  --epochs 30 --lr 1e-3 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v26_epochs *>&1 | Tee-Object -FilePath $log -Append

$best = python -c "import re;lines=open(r'$env:TEMP\mazlum_v26.log',encoding='utf-8',errors='ignore').read().splitlines();rows=[l for l in lines if l.startswith('epoch')];best=min(rows,key=lambda l: float(l.split('val')[1].split('/')[0]));print(best.split()[1].split('/')[0])"
Step "best val epoch: $best"

foreach ($ep in @($best, "30")) {
    $path = "$root\data\mazlum\v26_epochs\epoch_$('{0:D3}' -f [int]$ep).npz"
    $env:MAZLUM_MODEL = $path
    Step "v26 epoch $ep vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v26_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue

Step "done"
