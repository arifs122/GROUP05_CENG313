# v16: low-rank dataset from our own Kaggle matches, then self-critique fine-tune.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_v16_train.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

Step "build v16 dataset (637 new Kaggle episodes, Banette)"
python -u tools/mazlum/build_targeted_dataset.py --capacity 400000 `
  --episode-ids data/mazlum/v16_episode_ids.json --archetype Banette `
  --out data/mazlum/dataset_v16_kaggle_meta.npz 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "build FAILED"; exit 1 }

Step "fine-tune v15 -> v16 (lr 1e-4, 10 epochs, save-each)"
python -u src/mazlum/train_bc.py --dataset data/mazlum/dataset_v16_kaggle_meta.npz `
  --init data/mazlum/model_banette_v15.npz --out data/mazlum/model_v16_kaggle_meta.npz `
  --epochs 10 --lr 1e-4 --batch-size 64 --arena-every 0 --print-every 1 `
  --save-each data/mazlum/v16_epochs 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Step "train FAILED"; exit 1 }

Step "done"
