# v24: Slowking hard-matchup fine-tune (Dragapult + Alakazam) from v22 ep23.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v24.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v24 from v22 ep23 on Slowking vs Dragapult/Alakazam (lr 1e-4, 6 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v24_slowking_hard.npz `
  --init data/mazlum/v22_epochs/epoch_023.npz --out data/mazlum/model_v24_slowking_hard.npz `
  --epochs 6 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v24_epochs *>&1 | Tee-Object -FilePath $log -Append

Step "control: v22 ep23 vs dragapult (100 games, seed 2024)"
$env:MAZLUM_MODEL = "$root\data\mazlum\v22_epochs\epoch_023.npz"
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 100 --seed 2024 `
  --out results/v24_control_dragapult_100.json *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append

Step "screen v24 checkpoints vs dragapult (100 each, seed 2024)"
python -u tools/screen_checkpoints.py --dir data/mazlum/v24_epochs `
  --opponent "$root\bots\dragapult" --games 100 --seed 2024 `
  --out results/v24_screen.json *>&1 | Tee-Object -FilePath $log -Append

Step "best v24 vs banette (100 games, seed 2024)"
$best = python -c "import json;d=json.load(open('results/v24_screen.json',encoding='utf-8'));b=max(d['results'],key=lambda r:r['win_rate']);print(b['checkpoint'])"
$env:MAZLUM_MODEL = "$root\data\mazlum\v24_epochs\$best"
"best: $best" | Tee-Object -FilePath $log -Append
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/banette --games 100 --seed 2024 `
  --out "results/v24_${best}_banette_100.json" *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
