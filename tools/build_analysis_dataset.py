#!/usr/bin/env python3
"""Build the compact, git-ignored analysis table used by the notebooks.

The competition does not provide a ``train.csv``; the observed data are the
episode replays.  This script reads the lightweight metadata table produced by
``tools/build_master_replays.py`` and adds:

* parsed deck lengths and unique-card counts,
* card-kind composition per deck,
* whether each deck contains an ACE SPEC card,
* the player who went first (read from the beginning of each replay).

It writes ``data/analysis_matches.csv``.  The file is competition data and is
covered by ``.gitignore``; never commit it.

Usage:
    python tools/build_analysis_dataset.py
    python tools/build_analysis_dataset.py --master results/master_replays.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import MATCH_CSV, MASTER_CSV, build_match_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master",
        type=Path,
        default=MASTER_CSV,
        help="metadata CSV from tools/build_master_replays.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MATCH_CSV,
        help="output analysis CSV (default: data/analysis_matches.csv)",
    )
    parser.add_argument("--replays", type=Path, default=ROOT / "replays")
    args = parser.parse_args()

    if not args.master.is_file():
        raise SystemExit(
            f"metadata not found: {args.master}\n"
            "Build it first with:\n"
            f"  python tools/build_master_replays.py --out-dir {args.master.parent}"
        )

    table = build_match_table(args.master, args.replays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"wrote {args.out} ({len(table)} matches)")
    print("source groups:")
    print(table["source_group"].value_counts().to_string())
    print("\ntop archetypes:")
    print(table["arch0"].value_counts().head(10).to_string())
    print("\nfirst-player information:", table["first_player"].notna().sum(), "matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
