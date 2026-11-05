# v19 ep14: 1000-game confirmation vs dragapult and banette.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v19_confirm.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ckpt = "$root\data\mazlum\v19_epochs\epoch_014.npz"

foreach ($opp in @("dragapult", "banette")) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === ep14 vs $opp, 1000 games (seed 777) ===" | Tee-Object -FilePath $log -Append
    $env:MAZLUM_MODEL = $ckpt
    python -u tools/arena.py --bot0 bots/mazlum --bot1 "bots/$opp" --games 1000 --seed 777 `
      --out "results/v19_epoch_014_${opp}_1000.json" 2>&1 |
      Select-String -Pattern "^P0 wins|^P1 wins|^errors|^avg decision" | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\MAZLUM_MODEL
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === done ===" | Tee-Object -FilePath $log -Append
