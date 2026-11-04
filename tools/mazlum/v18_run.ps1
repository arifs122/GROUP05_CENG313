# v18 smart meta: fine-tune from v15, screen 10 checkpoints, run full battery.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v18_train.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v18 from v15 (lr 1e-4, 10 epochs, save-each)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v18_kaggle_smart.npz `
  --init data/mazlum/model_banette_v15.npz --out data/mazlum/model_v18_kaggle_smart.npz `
  --epochs 10 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v18_epochs 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "screen 10 checkpoints vs dragapult (200 games each)"
python -u tools/screen_checkpoints.py --dir data/mazlum/v18_epochs `
  --opponent "$root\bots\dragapult" --games 200 --seed 777 `
  --out results/v18_screen.json 2>&1 | Tee-Object -FilePath $log -Append

$best = python -c "import json;d=json.load(open('results/v18_screen.json',encoding='utf-8'));b=max(d['results'],key=lambda r:r['win_rate']);print(b['checkpoint'])"
Step "best checkpoint: $best"

Step "battery: banette / dragapult / lucario x 500 ($best)"
python -u tools/bot_battery.py --model "data/mazlum/v18_epochs/$best" --games 500 2>&1 |
  Select-String -Pattern "^===|^P0 wins|^opponent|^banette|^dragapult|^lucario|^wrote|^---|^====" |
  Tee-Object -FilePath $log -Append

Step "done"
