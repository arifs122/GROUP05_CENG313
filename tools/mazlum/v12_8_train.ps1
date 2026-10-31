# v12.8 "Devasa Mimari" — 6 blok x 8 bas x d128, 871k satir, sifirdan.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v12_8.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "ensure Dragapult deck"
Copy-Item bots\mazlum\deck_dragapult.csv bots\mazlum\deck.csv -Force

Step "arch check"
python -u src/mazlum/model.py 2>&1 | Tee-Object -FilePath $log -Append

Step "train v12.8 from scratch (871k rows, 6x8x128, epochs 40, patience 5, arena 200)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_dragapult_v127.npz `
  --out data/mazlum/model_v12_8.npz --epochs 40 --patience 5 `
  --antimeta-weight 1.0 --arena-games 200 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "quantize to int4 (600 KB budget)"
python tools/mazlum/quantize_model.py --inp data/mazlum/model_v12_8.npz `
  --out data/mazlum/model_v12_8_int4.npz 2>&1 | Tee-Object -FilePath $log -Append

Step "arena: 1000 single games vs Banette (int4 shipped weights)"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_v12_8_int4.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_8_int4_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append

Step "arena: 1000 single games vs Banette (fp16 reference)"
$env:MAZLUM_MODEL = "$root\data\mazlum\model_v12_8.npz"
python tools/arena.py --bot0 bots/mazlum --bot1 bots/banette --games 1000 `
  --out results/mazlum_v12_8_fp16_single_1000.json 2>&1 | Tee-Object -FilePath $log -Append
Remove-Item Env:\MAZLUM_MODEL

Step "done"
