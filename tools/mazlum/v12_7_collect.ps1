# v12.7 "Buyuk Veri" — asama 1: Dragapult episode madenciligi (hedef 3.200).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_collect_v12_7.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === collect dragapult (target 3200, teams 563, subs-per-team 8) ===" | Tee-Object -FilePath $log -Append

python -u tools/mazlum/collect_dragapult.py --target-episodes 3200 --teams 563 `
  --subs-per-team 8 --delay 0.8 --list-delay 0.4 2>&1 | Tee-Object -FilePath $log -Append

$code = $LASTEXITCODE
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] === collect finished (exit $code) ===" | Tee-Object -FilePath $log -Append
exit $code
