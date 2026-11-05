# v20: big brain (v19) fine-tuned on v15 targeted data, then screen + 1000-game confirm.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v20.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v20: big arch + v15 targeted data (lr 1e-4, 10 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v15_dragapult_focus.npz `
  --init data/mazlum/model_v19_big.npz --out data/mazlum/model_v20_big_targeted.npz `
  --arch big --fp16-save --epochs 10 --lr 1e-4 --batch-size 64 --arena-every 0 `
  --print-every 1 --save-each data/mazlum/v20_epochs *>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "screen 10 checkpoints vs dragapult"
python -u tools/screen_checkpoints.py --dir data/mazlum/v20_epochs `
  --opponent "$root\bots\dragapult" --games 200 --seed 777 `
  --out results/v20_screen.json *>&1 | Tee-Object -FilePath $log -Append

$best = python -c "import json;d=json.load(open('results/v20_screen.json',encoding='utf-8'));b=max(d['results'],key=lambda r:r['win_rate']);print(b['checkpoint'])"
Step "best checkpoint: $best"

foreach ($opp in @("dragapult", "banette")) {
    Step "confirm $best vs $opp (1000 games, seed 777)"
    $env:MAZLUM_MODEL = "$root\data\mazlum\v20_epochs\$best"
    python -u tools/arena.py --bot0 bots/mazlum --bot1 "bots/$opp" --games 1000 --seed 777 `
      --out "results/v20_${best}_${opp}_1000.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors|^avg decision" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL

Step "done"
