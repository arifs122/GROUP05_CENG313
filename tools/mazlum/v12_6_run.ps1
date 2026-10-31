# v12.6 "Sinirlari Kaldir": v12.3 data (385k, equal weight), epochs 100, patience 10.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v12_6.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "train v12.6 (385k mixed, antimeta weight 1.0, epochs 100, patience 10, arena 200)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult.npz `
  --out data/mazlum/model_v12_6.npz --epochs 100 --patience 10 `
  --antimeta-weight 1.0 --arena-games 200 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette (v12.6)"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_v12_6.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_6_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
