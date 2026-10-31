# v12.4.1: fine-tune the 73.5% v12.3 model with anti-meta weight 0.5 (5 epochs, lr 1e-4).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v1241.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "fine-tune v12.4.1 (init=model.npz v12.3, lr=1e-4, 5 epochs, anti-meta 0.5)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult.npz `
  --out data/mazlum/model_dragapult_v1241.npz --init data/mazlum/model.npz `
  --epochs 5 --lr 1e-4 --antimeta-weight 0.5 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_dragapult_v1241.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_dragapult_v1241_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
