# v42: fine-tune v21 on a 50/50 mix of general Banette and Banette-vs-Alakazam data.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v42.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "4"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v42 from v21 ep2 on 643k mix (4 epochs, lr 5e-5)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v42_mix.npz `
  --init data/mazlum/v21_epochs/epoch_002.npz --out data/mazlum/model_v42_alakazam.npz `
  --epochs 4 --lr 5e-5 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v42_epochs *>&1 | Tee-Object -FilePath $log -Append

foreach ($ep in @(2, 4)) {
    $file = "epoch_$('{0:D3}' -f $ep).npz"
    if (-not (Test-Path "$root\data\mazlum\v42_epochs\$file")) { continue }
    $env:MAZLUM_MODEL = "$root\data\mazlum\v42_epochs\$file"
    Step "v42 $file vs public alakazam (100 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum --bot1 bots/public_alakazam --games 100 --seed 777 `
      --out "results/v42_ep${ep}_alakazam_100.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
    Step "v42 $file vs dragapult (100 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum --bot1 bots/dragapult --games 100 --seed 777 `
      --out "results/v42_ep${ep}_dragapult_100.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
