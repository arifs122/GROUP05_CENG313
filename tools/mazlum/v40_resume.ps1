# v40 resume: continue Dragapult training from the last checkpoint.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v40_resume.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "4"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

$last = Get-ChildItem data/mazlum/v40_epochs -Filter "epoch_*.npz" | Sort-Object Name | Select-Object -Last 1
$start = [int]($last.BaseName -replace "epoch_", "")
$remaining = 25 - $start
Step "resume v40 from $($last.Name) ($remaining epochs left)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v40_dragapult_win.npz `
  --arch big --init $last.FullName --out data/mazlum/model_v40_dragapult.npz `
  --epochs $remaining --lr 5e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v40b_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(5, 10, 17)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v40b_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v40b_epochs\$file"
    Step "v40b $file vs dragapult bot (mirror, 300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_dragapult --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v40b_ep${ep}_mirror_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
