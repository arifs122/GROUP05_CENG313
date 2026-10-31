# Dragapult data scaling: collect -> build -> train v12.1 -> 1000-game arena.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_scale.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "collect Dragapult episodes (target 1800, backoff, 1 worker)"
python tools/mazlum/collect_dragapult.py --target-episodes 1800 --teams 400 --workers 1 --delay 1.0 `
  2>&1 | Tee-Object -FilePath $log -Append

Step "build Dragapult dataset from replays"
python tools/mazlum/build_dragapult_dataset.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "build FAILED"; exit 1 }

Step "train v12.1 (4 blocks, 4 heads, Pre-LN, FP16) on the scaled set"
python src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult.npz `
  --out data/mazlum/model_dragapult_v121.npz 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_dragapult_v121.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_dragapult_scaled_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "promote v12.1 to model.npz"
Copy-Item data\mazlum\model_dragapult_v121.npz data\mazlum\model.npz -Force

Step "done"
