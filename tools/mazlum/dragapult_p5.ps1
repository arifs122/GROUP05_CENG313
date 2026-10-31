# Dragapult early-stopping repair: patience 5, 200-game mini arena, 1000-game final.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_patience5.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "train v12.1 (patience 5, arena 200, epochs<=40)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult.npz `
  --out data/mazlum/model_dragapult_p5.npz --epochs 30 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_dragapult_p5.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_dragapult_p5_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
