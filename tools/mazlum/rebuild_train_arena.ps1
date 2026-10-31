# Rebuild the dataset from all mined replays, retrain, then measure 1000 games.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_rebuild.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "build dataset (all replays)"
python src/mazlum/build_dataset.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "build FAILED (exit $LASTEXITCODE)"; exit 1 }

Step "train (arena >= + val top1 tie-break)"
python src/mazlum/train_bc.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED (exit $LASTEXITCODE)"; exit 1 }

Step "arena: 1000 single games"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v10_1800ep_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append

Step "done"
