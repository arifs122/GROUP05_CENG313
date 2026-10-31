"""Screen per-epoch checkpoints against one opponent and print a table.

Usage::

    python tools/screen_checkpoints.py --dir data/mazlum/v15_epochs \
        --opponent C:\\...\\bots\\dragapult --games 200 --seed 777
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen checkpoints against an opponent")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--bot0", type=Path, default=ROOT / "bots" / "mazlum")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    checkpoints = sorted(args.dir.glob("epoch_*.npz"))
    if not checkpoints:
        print(f"no checkpoints in {args.dir}")
        return 1

    rows = []
    for ckpt in checkpoints:
        result_path = args.dir / f"_screen_{ckpt.stem}.json"
        env = dict(os.environ, MAZLUM_MODEL=str(ckpt.resolve()), PYTHONIOENCODING="utf-8")
        cmd = [
            sys.executable, "-u", str(ROOT / "tools" / "arena.py"),
            "--bot0", str(args.bot0),
            "--bot1", str(args.opponent),
            "--games", str(args.games),
            "--seed", str(args.seed),
            "--out", str(result_path),
        ]
        proc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode != 0 or not result_path.is_file():
            rows.append((ckpt.name, None))
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        wins, losses, draws = data["wins"]
        rows.append((ckpt.name, wins / data["games"]))
        result_path.unlink(missing_ok=True)
        print(f"{ckpt.name}: {wins}/{data['games']} = {100 * wins / data['games']:.1f}%", flush=True)

    print("\ncheckpoint            win rate")
    print("-" * 34)
    for name, rate in rows:
        print(f"{name:<22}{'-' if rate is None else f'{100 * rate:5.1f}%':>10}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"games": args.games, "opponent": str(args.opponent),
                        "results": [{"checkpoint": n, "win_rate": r} for n, r in rows]}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
