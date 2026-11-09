#!/usr/bin/env python3
"""Extract the Top1 Mega Kangaskhan/Ogerpon list from winning replays."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

import replay_meta as rm  # noqa: E402

TEAM_ID = "17014244"
TEAM_NAME = "やる気元気ミワハルキ 2nd ver."


def main() -> int:
    base = ROOT / "replays" / "top5_nov1" / TEAM_ID
    variants: Counter = Counter()
    examples: dict[tuple, str] = {}
    games = 0
    for path in sorted(base.glob("*.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = rm.team_names(episode)[:2]
        ours = [i for i, n in enumerate(names) if TEAM_NAME in str(n)]
        if len(ours) != 1:
            continue
        me = ours[0]
        if rm.winner_index(episode) != me:
            continue
        decks = rm.extract_decks(episode)
        if not decks:
            continue
        key = tuple(sorted(Counter(decks[me]).items()))
        variants[key] += 1
        examples.setdefault(key, path.name)
        games += 1

    if not variants:
        raise SystemExit("no winning replays found")
    key, count = variants.most_common(1)[0]
    deck: list[int] = []
    for cid, n in key:
        deck.extend([cid] * n)
    print(f"games={games} distinct_lists={len(variants)} chosen={count} example={examples[key]}")
    print(f"cards={len(deck)} unique={len(set(deck))} archetype={rm.archetype(deck)}")
    for cid, n in key:
        print(f"  {n:2d} x {cid:4d} {rm.card_name(cid)}")

    out = ROOT / "agents" / "mk_ogrepon" / "deck_top1.csv"
    out.write_text("\n".join(map(str, deck)) + "\n", encoding="utf-8")
    (ROOT / "agents" / "mk_ogrepon" / "bot" / "deck.csv").write_text(
        "\n".join(map(str, deck)) + "\n", encoding="utf-8")
    print(f"written {out} and bot/deck.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
