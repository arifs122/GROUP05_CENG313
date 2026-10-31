# Mazlum nightly loop: collect -> build -> train -> 1000-game arena.
# Run with:  powershell -NoProfile -ExecutionPolicy Bypass -File tools\mazlum\nightly.ps1
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_nightly.log"

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "collect: target 1800, teams 491, workers 1, delay 1.0 (backoff on 429)"
python tools/mazlum/collect_replays.py --target 1800 --teams 491 --workers 1 --delay 1.0 `
  --archetypes "Banette,Alakazam,Dragapult,Festival Lead,Hydrapple,Mega Kangaskhan,Mega Abomasnow,Mega Excadrill,Ceruledge,Mega Charizard,Mega Gardevoir,Mega Lucario,Chandelure,Crustle,Mega Starmie,Slowking-Kangaskhan,Marnie's Grimmsnarl,Cynthia's Garchomp,Archaludon,Iono's Bellibolt,Mega Lopunny,Mega Venusaur,Mega Gengar,Mega Absol,Mega Blaziken,Ethan's Typhlosion,Hop's Trevenant" `
  2>&1 | Tee-Object -FilePath $log -Append

Step "build dataset"
python src/mazlum/build_dataset.py 2>&1 | Tee-Object -FilePath $log -Append

Step "train (arena early stopping >=, val top1 tie-break)"
python src/mazlum/train_bc.py 2>&1 | Tee-Object -FilePath $log -Append

Step "arena: 1000 single games"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v10_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append

Step "done"
