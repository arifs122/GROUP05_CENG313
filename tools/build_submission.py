#!/usr/bin/env python3
"""Build a Kaggle submission bundle.

The bundle layout is::

    main.py
    policy.py
    deck.csv
    cg/...

Usage:
    python3 tools/build_submission.py
    python3 tools/build_submission.py --deck decks/mega_abomasnow.csv
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MIB = 197.7


def main() -> int:
    parser = argparse.ArgumentParser(description="Build submission.tar.gz")
    parser.add_argument("--deck", type=Path, default=ROOT / "decks" / "mega_abomasnow.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "submissions" / "submission.tar.gz")
    parser.add_argument("--stage", type=Path, default=ROOT / "build" / "submission")
    args = parser.parse_args()

    agent_dir = ROOT / "src" / "agent"
    cg_dir = ROOT / "vendor" / "cg"

    if not args.deck.is_file():
        raise SystemExit(f"deck not found: {args.deck}")
    if not (agent_dir / "main.py").is_file():
        raise SystemExit(f"agent entry point not found: {agent_dir / 'main.py'}")
    if not cg_dir.is_dir():
        raise SystemExit(
            f"engine not found: {cg_dir}\n"
            "Run: bash scripts/setup_assets.sh /path/to/sample_submission/cg"
        )

    if args.stage.exists():
        shutil.rmtree(args.stage)
    args.stage.mkdir(parents=True)

    for path in sorted(agent_dir.glob("*.py")):
        shutil.copy2(path, args.stage / path.name)
    shutil.copy2(args.deck, args.stage / "deck.csv")
    shutil.copytree(cg_dir, args.stage / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    with tarfile.open(args.out, "w:gz") as tar:
        for path in sorted(args.stage.rglob("*")):
            tar.add(path, arcname=path.relative_to(args.stage), recursive=False)

    size_mib = args.out.stat().st_size / (1024 * 1024)
    with tarfile.open(args.out, "r:gz") as tar:
        top = sorted({name.split("/")[0] for name in tar.getnames()})

    print(f"created {args.out} ({size_mib:.2f} MiB)")
    print(f"top-level entries: {', '.join(top)}")
    if size_mib > LIMIT_MIB:
        print(f"WARNING: bundle exceeds the {LIMIT_MIB} MiB competition limit")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
