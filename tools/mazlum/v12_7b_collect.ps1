# v12.7 "Buyuk Veri" — asama 1b: mirror acigini kapatmak icin 3.900 episode.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_collect_v12_7b.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === collect dragapult round 2 (target 3900) ===" | Tee-Object -FilePath $log -Append

python -u tools/mazlum/collect_dragapult.py --target-episodes 3900 --teams 563 `
  --subs-per-team 8 --delay 0.8 --list-delay 0.4 2>&1 | Tee-Object -FilePath $log -Append

$code = $LASTEXITCODE
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === collect finished (exit $code) ===" | Tee-Object -FilePath $log -Append
exit $code
