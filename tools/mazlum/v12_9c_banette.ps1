# v12.9c — Banette fine-tune with v12.3's mixture ratio (anti-meta weight 0.5).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v12_9c.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "fine-tune v12.3 -> Banette-only, anti-meta 0.5 (2.9:1 ratio), corrected cadence"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_banette_v123.npz `
  --init data/mazlum/model.npz --out data/mazlum/model_v12_9c_banette_ft.npz `
  --epochs 12 --patience 2 --arena-every 3 --print-every 3 --arena-games 200 `
  --lr 1e-4 --batch 64 --antimeta-weight 0.5 --device cpu 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "fine-tune FAILED"; exit 1 }

Step "arena: 1000 single games vs Banette (v12.9c)"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_v12_9c_banette_ft.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_9c_banette_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
