# v19 big brain: 4x4 d=80 Pre-LN over 37-dim state, cold start on v17 data.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v19.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v19 cold start (big arch, v17 data, lr 3e-4, 15 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v17_dragapult_massive.npz `
  --out data/mazlum/model_v19_big.npz --arch big --fp16-save `
  --epochs 15 --lr 3e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v19_epochs *>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "screen 15 checkpoints vs dragapult"
python -u tools/screen_checkpoints.py --dir data/mazlum/v19_epochs `
  --opponent "$root\bots\dragapult" --games 200 --seed 777 `
  --out results/v19_screen.json *>&1 | Tee-Object -FilePath $log -Append

$best = python -c "import json;d=json.load(open('results/v19_screen.json',encoding='utf-8'));b=max(d['results'],key=lambda r:r['win_rate']);print(b['checkpoint'])"
Step "best checkpoint: $best"

Step "battery: banette / dragapult / lucario x 500 ($best)"
python -u tools/bot_battery.py --model "data/mazlum/v19_epochs/$best" --games 500 2>&1 |
  Select-String -Pattern "^===|^P0 wins|^opponent|^banette|^dragapult|^lucario|^wrote|^---|^====" |
  Tee-Object -FilePath $log -Append

Step "done"
