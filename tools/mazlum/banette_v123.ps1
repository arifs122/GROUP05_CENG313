# Banette champion run with the v12.3 stack (4 blocks, Pre-LN, FP16, patience 5).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_banette_v123.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Banette deck"
Copy-Item bots\mazlum\deck_banette.csv bots\mazlum\deck.csv -Force

Step "train v12.3 on the 688k Banette dataset (patience 5, arena 200)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset.npz `
  --out data/mazlum/model_banette_v123.npz --epochs 30 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette bot"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_banette_v123.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_banette_v123_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
