# v25: 300-game confirmation of v24 + mixed fine-tune (full + hard duplicated).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v25.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

$env:MAZLUM_MODEL = "$root\data\mazlum\v22_epochs\epoch_023.npz"
Step "control v22 ep23 vs dragapult (300 games, seed 777)"
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
  --out results/v25_control_dragapult_300.json *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append

$env:MAZLUM_MODEL = "$root\data\mazlum\v24_epochs\epoch_002.npz"
Step "v24 ep2 vs dragapult (300 games, seed 777)"
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
  --out results/v25_v24ep2_dragapult_300.json *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append

Step "train v25 from v24 ep2 on mixed data (lr 5e-5, 4 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v25_mix.npz `
  --init data/mazlum/v24_epochs/epoch_002.npz --out data/mazlum/model_v25_mix.npz `
  --epochs 4 --lr 5e-5 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v25_epochs *>&1 | Tee-Object -FilePath $log -Append

Step "screen v25 checkpoints vs dragapult (100 each, seed 2024)"
python -u tools/screen_checkpoints.py --dir data/mazlum/v25_epochs `
  --opponent "$root\bots\dragapult" --games 100 --seed 2024 `
  --out results/v25_screen.json *>&1 | Tee-Object -FilePath $log -Append

Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
