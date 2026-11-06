# Resume v36 (wide arch) from the latest per-epoch checkpoint, then evaluate.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v36_resume.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "4"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

$last = Get-ChildItem data/mazlum/v36_epochs -Filter "epoch_*.npz" -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -Last 1
if (-not $last) {
    Step "no v36 checkpoint found; starting from scratch"
    $initArgs = @()
    $remaining = 25
    $startEpoch = 0
} else {
    $startEpoch = [int]($last.BaseName -replace "epoch_", "")
    $remaining = 25 - $startEpoch
    Step "resuming v36 from $($last.Name) ($remaining epochs left)"
    $initArgs = @("--init", $last.FullName)
}

if ($remaining -gt 0) {
    python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v34_v2_slowking.npz `
      --arch wide @initArgs --out data/mazlum/model_v36_wide.npz `
      --epochs $remaining --lr 5e-4 --batch-size 64 --arena-every 0 --print-every 1 `
      --save-each data/mazlum/v36b_epochs *>&1 | Tee-Object -FilePath $log -Append
}

foreach ($ep in @(5, 10, 15, 21)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v36b_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v36b_epochs\$file"
    Step "v36b $file vs dragapult (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/dragapult --games 300 --seed 777 `
      --out "results/v36b_ep${ep}_dragapult_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
