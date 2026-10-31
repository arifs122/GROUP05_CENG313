# Stage the v12.7 dataset + code for `kaggle datasets create`.
$ErrorActionPreference = "Stop"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
$stage = "C:\Users\Fariz\AppData\Local\Temp\opencode\kaggle_ds\mazlum-dragapult-v127"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path "$stage\code\src\mazlum" | Out-Null
New-Item -ItemType Directory -Path "$stage\code\tools" | Out-Null
New-Item -ItemType Directory -Path "$stage\code\bots" | Out-Null

Write-Output "copying dataset..."
Copy-Item "$root\data\mazlum\dataset_dragapult_v127.npz" "$stage\dataset_dragapult_v127.npz"
Copy-Item "$root\train_kaggle.ipynb" "$stage\train_kaggle.ipynb"

Write-Output "copying code..."
Copy-Item "$root\src\mazlum\model.py"        "$stage\code\src\mazlum\"
Copy-Item "$root\src\mazlum\train_bc.py"     "$stage\code\src\mazlum\"
Copy-Item "$root\src\mazlum\features.py"     "$stage\code\src\mazlum\"
Copy-Item "$root\src\mazlum\quantize.py"     "$stage\code\src\mazlum\"
Copy-Item "$root\tools\arena.py"             "$stage\code\tools\"
Copy-Item "$root\tools\mazlum\quantize_model.py" "$stage\code\tools\"
Copy-Item "$root\vendor\cg"                  "$stage\code\vendor\cg" -Recurse
Copy-Item "$root\bots\mazlum"                "$stage\code\bots\mazlum" -Recurse
Copy-Item "$root\bots\banette"               "$stage\code\bots\banette" -Recurse

# Drop bytecode / local artefacts.
Get-ChildItem $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

$files = Get-ChildItem $stage -Recurse -File
$size = ($files | Measure-Object -Property Length -Sum).Sum
Write-Output ("staged {0} files, {1:N1} MB -> {2}" -f $files.Count, ($size / 1MB), $stage)
