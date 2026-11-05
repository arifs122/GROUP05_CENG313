# v21: matchup-targeted fine-tune (Hydrapple/Archaludon/Festival/Grimmsnarl/Lopunny) from v15.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v21.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "train v21 from v15 on matchup data (lr 1e-4, 8 epochs)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v21_matchups.npz `
  --init data/mazlum/model_banette_v15.npz --out data/mazlum/model_v21_matchups.npz `
  --epochs 8 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v21_epochs *>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "screen 8 checkpoints vs dragapult (regression guard)"
python -u tools/screen_checkpoints.py --dir data/mazlum/v21_epochs `
  --opponent "$root\bots\dragapult" --games 200 --seed 777 `
  --out results/v21_screen.json *>&1 | Tee-Object -FilePath $log -Append

$best = python -c "import json;d=json.load(open('results/v21_screen.json',encoding='utf-8'));b=max(d['results'],key=lambda r:r['win_rate']);print(b['checkpoint'])"
Step "best checkpoint: $best"

foreach ($opp in @("dragapult", "banette")) {
    Step "confirm $best vs $opp (1000 games, seed 777)"
    $env:MAZLUM_MODEL = "$root\data\mazlum\v21_epochs\$best"
    python -u tools/arena.py --bot0 bots/mazlum --bot1 "bots/$opp" --games 1000 --seed 777 `
      --out "results/v21_${best}_${opp}_1000.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors|^avg decision" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL

Step "done"
