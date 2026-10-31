"""Bot battery: run the model against several opponents, 500 games each.

Each match-up runs in its own ``arena.py`` subprocess so the opponent's
``features``/logging cannot leak into our agent process.

Usage::

    python tools/bot_battery.py --model data/mazlum/model_v14_multi.npz --games 500
    python tools/bot_battery.py --model ... --bots bots/banette bots/dragapult --games 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOTS = [
    ROOT / "bots" / "banette",
    ROOT / "bots" / "dragapult",
    ROOT / "bots" / "lucario",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model against a battery of opponents")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bot0", type=Path, default=ROOT / "bots" / "mazlum")
    parser.add_argument("--bots", type=Path, nargs="+", default=DEFAULT_BOTS)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.is_file():
        print(f"error: model not found: {model}")
        return 1

    stamp = model.stem
    out = args.out or ROOT / "results" / f"battery_{stamp}_{args.games}.json"
    rows = []
    env = dict(os.environ, MAZLUM_MODEL=str(model), PYTHONIOENCODING="utf-8")

    for opponent in args.bots:
        opponent = opponent.resolve()
        if not (opponent / "main.py").is_file():
            print(f"skip (no main.py): {opponent}")
            continue
        result_path = out.with_name(f"_tmp_{opponent.name}_{args.games}.json")
        cmd = [
            sys.executable, "-u", str(ROOT / "tools" / "arena.py"),
            "--bot0", str(args.bot0),
            "--bot1", str(opponent),
            "--games", str(args.games),
            "--seed", str(args.seed),
            "--out", str(result_path),
        ]
        print(f"\n=== {opponent.name}: {args.games} games ===", flush=True)
        proc = subprocess.run(cmd, env=env, capture_output=False)
        if proc.returncode != 0 or not result_path.is_file():
            print(f"  failed (exit {proc.returncode})")
            rows.append({"opponent": opponent.name, "wins": 0, "games": 0, "win_rate": None, "error": True})
            continue

        data = json.loads(result_path.read_text(encoding="utf-8"))
        wins, losses, draws = data["wins"]
        games = data["games"]
        rows.append(
            {
                "opponent": opponent.name,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": games,
                "win_rate": wins / games if games else None,
                "errors": data.get("errors", 0),
            }
        )
        result_path.unlink(missing_ok=True)

    print("\n" + "=" * 52)
    print(f"{'opponent':<22}{'wins':>8}{'games':>8}{'win rate':>11}")
    print("-" * 52)
    for row in rows:
        rate = "-" if row["win_rate"] is None else f"{100 * row['win_rate']:.1f}%"
        print(f"{row['opponent']:<22}{row['wins']:>8}{row['games']:>8}{rate:>11}")
    print("=" * 52)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": str(model), "games": args.games, "results": rows}, indent=2), encoding="utf-8")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
