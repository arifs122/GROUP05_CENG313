# v12: 4-block / 4-head FP16 model — train then measure 1000 games.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v12.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v12 (4 blocks x 4 heads, FP16 export, arena >= + val tie-break)"
python src/mazlum/train_bc.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED (exit $LASTEXITCODE)"; exit 1 }

Step "arena: 1000 single games"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_fp16_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append

Step "done"
