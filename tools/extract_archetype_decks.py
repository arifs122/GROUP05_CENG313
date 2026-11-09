#!/usr/bin/env python3
"""Extract exact deck lists for an archetype from ``results/master_replays.csv``.

The replay corpus already stores the initial 60-card list for both seats and
the archetype classifier output.  This tool groups identical lists and writes
the top-N lists to CSV, with a small summary (games, win rate, pilots, sources).

Examples
--------
    python3 tools/extract_archetype_decks.py \
        --archetype "Mega Excadrill" --top 5 \
        --out decks/ppo_candidates --prefix mega_excadrill

    python3 tools/extract_archetype_decks.py \
        --archetype Dragapult --top 5 --source top20 \
        --out /tmp/opencode/drag_lists --prefix dragapult_top20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "results" / "master_replays.csv"
CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "deck"


def load_card_names() -> dict[int, str]:
    names: dict[int, str] = {}
    if not CARD_CSV.exists():
        return names
    with CARD_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                names[int(row["Card ID"])] = row["Card Name"]
            except (KeyError, ValueError):
                pass
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetype", required=True, help="exact arch0/arch1 label")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--source", default=None, help="optional source filter, e.g. top20")
    parser.add_argument("--out", type=Path, default=ROOT / "decks" / "ppo_candidates")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--master", type=Path, default=MASTER)
    args = parser.parse_args()

    if not args.master.exists():
        raise SystemExit(f"{args.master} not found; run tools/build_master_replays.py first")

    prefix = args.prefix or _slug(args.archetype)
    stats: dict[tuple[int, ...], dict] = defaultdict(
        lambda: {"games": 0, "wins": 0, "teams": Counter(), "sources": Counter()}
    )

    with args.master.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if args.source and row.get("source") != args.source:
                continue
            for arch, deck, side, team in (
                (row.get("arch0"), row.get("deck0"), "0", row.get("team0")),
                (row.get("arch1"), row.get("deck1"), "1", row.get("team1")),
            ):
                if arch != args.archetype or not deck:
                    continue
                try:
                    cards = tuple(sorted(int(x) for x in deck.split()))
                except ValueError:
                    continue
                if len(cards) != 60:
                    continue
                entry = stats[cards]
                entry["games"] += 1
                winner = row.get("winner")
                if winner in ("0", "1"):
                    entry["wins"] += winner == side
                entry["teams"][team or "?"] += 1
                entry["sources"][row.get("source") or "?"] += 1

    ranked = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["games"], -(kv[1]["wins"] / max(1, kv[1]["games"]))),
    )
    ranked = [(cards, st) for cards, st in ranked if st["games"] >= args.min_games]
    if not ranked:
        raise SystemExit(f"no decks found for archetype {args.archetype!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    names = load_card_names()
    summary = []
    for rank, (cards, st) in enumerate(ranked[: args.top], 1):
        path = args.out / f"{prefix}_{rank}.csv"
        path.write_text("\n".join(map(str, cards)) + "\n", encoding="utf-8")
        wr = st["wins"] / st["games"] if st["games"] else 0.0
        top_cards = ", ".join(
            f"{n}x {names.get(cid, f'#{cid}')}"
            for cid, n in Counter(cards).most_common(12)
        )
        record = {
            "rank": rank,
            "file": str(path),
            "games": st["games"],
            "wins": st["wins"],
            "win_rate": wr,
            "teams": dict(st["teams"].most_common(5)),
            "sources": dict(st["sources"].most_common(5)),
            "top_cards": top_cards,
        }
        summary.append(record)
        print(f"\n#{rank} {path}  games={st['games']}  wr={wr:.1%}")
        print(f"   teams: {', '.join(f'{k[:30]}({v})' for k, v in st['teams'].most_common(3))}")
        print(f"   cards: {top_cards}")

    summary_path = args.out / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
