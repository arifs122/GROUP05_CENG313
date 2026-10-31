# Champion re-measurement: v12.3 (model.npz) vs Banette, fresh seed + shuffled turn order.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v123_recheck.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === v12.3 re-check: 1000 games, seed 20261029, shuffled first player ===" | Tee-Object -FilePath $log -Append

$env:MAZLUM_MODEL = "$root\data\mazlum\model.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --seed 20261029 --shuffle `
  --out results/mazlum_v12_3_seed20261029_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === done ===" | Tee-Object -FilePath $log -Append
