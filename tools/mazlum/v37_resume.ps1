# v37 resume: finish the last 5 epochs from ep20 and evaluate ep25.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v37_resume.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "3"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "resume v37 from ep20 (5 epochs, lr 5e-4)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v37_v2_slowking.npz `
  --arch big --init data/mazlum/v37_epochs/epoch_020.npz --out data/mazlum/model_v37_v2.npz `
  --epochs 5 --lr 5e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v37b_epochs *>&1 | Tee-Object -FilePath $log -Append

$env:MAZLUM_MODEL = "$root\data\mazlum\v37b_epochs\epoch_005.npz"
Step "v37b ep25 vs dragapult (300 games, seed 777)"
python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
  --out "results/v37b_ep25_dragapult_300.json" *>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
