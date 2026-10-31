# v12.7 "Buyuk Veri" — asama 2: derle, sifirdan egit (40 epoch), 1000 mac arena.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v12_7_train.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "build dataset (v12.7, all episodes, season 2 pool)"
python -u tools/mazlum/build_dragapult_dataset.py `
  --out data/mazlum/dataset_dragapult_v127.npz 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "build FAILED"; exit 1 }

Step "train v12.7 from scratch (equal weight 1.0, epochs 40, patience 5, arena 200)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult_v127.npz `
  --out data/mazlum/model_v12_7.npz --epochs 40 --patience 5 `
  --antimeta-weight 1.0 --arena-games 200 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette (v12.7)"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_v12_7.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_7_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
