#!/usr/bin/env python3
"""Build a Kaggle submission bundle.

Bundle layout::

    main.py
    <bot helper files>
    deck.csv
    cg/...

Usage:
    python3 tools/build_submission.py
    python3 tools/build_submission.py --bot bots/benim_botum
    python3 tools/build_submission.py --deck decks/mega_abomasnow.csv
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MIB = 197.7


def read_deck_count(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a submission .tar.gz")
    parser.add_argument("--bot", type=Path, default=None,
                        help="bot folder (uses its main.py and deck.csv)")
    parser.add_argument("--deck", type=Path, default=None,
                        help="deck override (default: bot deck or decks/mega_abomasnow.csv)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stage", type=Path, default=ROOT / "build" / "submission")
    args = parser.parse_args()

    cg_dir = ROOT / "vendor" / "cg"
    if not cg_dir.is_dir():
        raise SystemExit(
            f"engine not found: {cg_dir}\n"
            "Run: bash scripts/setup_assets.sh /path/to/sample_submission/cg"
        )

    if args.bot is not None:
        bot_dir = args.bot.resolve()
        main_path = bot_dir / "main.py"
        deck_path = args.deck or (bot_dir / "deck.csv")
        if not bot_dir.is_dir():
            raise SystemExit(f"bot folder not found: {bot_dir}")
        if not main_path.is_file():
            raise SystemExit(f"bot entry point not found: {main_path}")
        if not deck_path.is_file():
            raise SystemExit(f"deck not found: {deck_path}")
        # Copy flat runtime files; skip metadata and documentation.
        source_files = [
            path
            for path in sorted(bot_dir.iterdir())
            if path.is_file()
            and path.name not in {"meta.json", "README.md", "readme.md"}
            and path.suffix not in {".pyc", ".md"}
            and not path.name.startswith(".")
        ]
        bundle_name = bot_dir.name
    else:
        bot_dir = None
        main_path = ROOT / "src" / "agent" / "main.py"
        deck_path = args.deck or (ROOT / "decks" / "mega_abomasnow.csv")
        if not main_path.is_file():
            raise SystemExit(f"agent entry point not found: {main_path}")
        if not deck_path.is_file():
            raise SystemExit(f"deck not found: {deck_path}")
        source_files = sorted((ROOT / "src" / "agent").glob("*.py"))
        bundle_name = "submission"

    deck_count = read_deck_count(deck_path)
    if deck_count != 60:
        raise SystemExit(f"{deck_path}: expected 60 cards, found {deck_count}")

    out_path = args.out or (ROOT / "submissions" / f"{bundle_name}.tar.gz")
    stage = args.stage

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for path in source_files:
        if path.name == "deck.csv":
            continue
        shutil.copy2(path, stage / path.name)
    shutil.copy2(deck_path, stage / "deck.csv")
    shutil.copytree(cg_dir, stage / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with tarfile.open(out_path, "w:gz") as tar:
        for path in sorted(stage.rglob("*")):
            tar.add(path, arcname=path.relative_to(stage), recursive=False)

    size_mib = out_path.stat().st_size / (1024 * 1024)
    with tarfile.open(out_path, "r:gz") as tar:
        top = sorted({name.split("/")[0] for name in tar.getnames()})

    print(f"created {out_path} ({size_mib:.2f} MiB)")
    print(f"bot: {bundle_name}")
    print(f"top-level entries: {', '.join(top)}")
    if size_mib > LIMIT_MIB:
        print(f"WARNING: bundle exceeds the {LIMIT_MIB} MiB competition limit")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
