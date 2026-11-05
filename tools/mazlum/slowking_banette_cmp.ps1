# Compare v22 vs v26 vs Banette bot (general strength check), same seed.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_slowking_banette_cmp.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

foreach ($pair in @(
    @{name = "v22_ep23"; path = "$root\data\mazlum\v22_epochs\epoch_023.npz"},
    @{name = "v26_ep30"; path = "$root\data\mazlum\v26_epochs\epoch_030.npz"}
)) {
    $env:MAZLUM_MODEL = $pair.path
    Step "$($pair.name) vs banette (300 games, seed 777)"
    python -u tools/arena.py --bot0 bots/mazlum_slowking --bot1 bots/banette --games 300 --seed 777 `
      --out "results/$($pair.name)_banette_300.json" *>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
