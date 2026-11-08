# Round-robin: candidates vs a fixed battery (dragapult, public_alakazam, public_meta).
$ErrorActionPreference = "Continue"
$root = "C:\Users\Fariz\OneDrive\Desktop\TransforMew"
Set-Location $root
$log = Join-Path $env:TEMP "mazlum_roundrobin.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string]$name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append
}

$candidates = @(
    @{name = "v15"; model = "$root\data\mazlum\model.npz"; bot = "bots/mazlum"},
    @{name = "v21"; model = "$root\data\mazlum\v21_epochs\epoch_002.npz"; bot = "bots/mazlum"},
    @{name = "v41"; model = "$root\data\mazlum\v41_epochs\epoch_025.npz"; bot = "bots/mazlum"},
    @{name = "dipam"; model = ""; bot = "bots/public_dipam"},
    @{name = "alakazam"; model = ""; bot = "bots/public_alakazam"},
    @{name = "grimmsnarl"; model = ""; bot = "bots/public_grimmsnarl"},
    @{name = "archaludon"; model = ""; bot = "bots/public_meta"}
)
$battery = @(
    @{name = "dragapult"; bot = "bots/dragapult"},
    @{name = "alakazam"; bot = "bots/public_alakazam"},
    @{name = "archaludon"; bot = "bots/public_meta"}
)

foreach ($c in $candidates) {
    foreach ($b in $battery) {
        if ($c.bot -eq $b.bot) { continue }
        if ($c.model -ne "") { $env:MAZLUM_MODEL = $c.model } else { Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue }
        Step "$($c.name) vs $($b.name) (20 games, seed 777)"
        python -u tools/arena.py --bot0 $c.bot --bot1 $b.bot --games 20 --seed 777 `
          --out "results/rr_$($c.name)_vs_$($b.name).json" *>&1 |
          Select-String -Pattern "^P0 wins|^P1 wins|^errors" | Tee-Object -FilePath $log -Append
    }
}
Remove-Item Env:\MAZLUM_MODEL -ErrorAction SilentlyContinue
Step "done"
