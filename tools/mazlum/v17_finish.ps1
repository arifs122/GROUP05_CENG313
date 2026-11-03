# v17 finish: confirm epoch_006 over 1000 games, then full battery.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v17_finish.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

$ckpt = "$root\data\mazlum\v17_epochs\epoch_006.npz"

Step "confirm epoch_006 vs dragapult (1000 games, seed 777)"
$env:MAZLUM_MODEL = $ckpt
python -u tools/arena.py --bot0 bots/mazlum --bot1 bots/dragapult --games 1000 --seed 777 `
  --out results/v17_epoch_006_dragapult_1000.json 2>&1 |
  Select-String -Pattern "^P0 wins|^P1 wins|^errors|^avg decision" | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "battery: banette / dragapult / lucario x 500"
python -u tools/bot_battery.py --model $ckpt --games 500 2>&1 |
  Select-String -Pattern "^===|^P0 wins|^opponent|^banette|^dragapult|^lucario|^wrote|^---|^====" |
  Tee-Object -FilePath $log -Append

Step "done"
