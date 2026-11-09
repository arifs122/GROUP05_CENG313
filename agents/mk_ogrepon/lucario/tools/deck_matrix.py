#!/usr/bin/env python3
"""Archetype deck-vs-deck matrix from the local replay corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as rm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, default=ROOT / "replays")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "agents" / "mk_ogrepon" / "lucario" / "reports" / "deck_matrix.json")
    args = parser.parse_args()

    shapes: Counter = Counter()
    matrix = defaultdict(lambda: [0, 0])  # (a,b) -> [a wins, b wins]
    episodes = 0
    seen = set()
    for path in args.replays.rglob("*.json"):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        eid = (e.get("info") or {}).get("EpisodeId") or e.get("id")
        if eid in seen:
            continue
        seen.add(eid)
        decks = rm.extract_decks(e)
        winner = rm.winner_index(e)
        if not decks or winner is None:
            continue
        episodes += 1
        a = rm.archetype(decks[0])
        b = rm.archetype(decks[1])
        shapes[a] += 1
        shapes[b] += 1
        matrix[(a, b)][0] += 1 if winner == 0 else 0
        matrix[(a, b)][1] += 1 if winner == 1 else 0

    rows = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (a, b), (aw, bw) in matrix.items():
        rows[a][b][0] += aw
        rows[a][b][1] += bw

    def rate(a, b):
        aw, bw = rows[a][b]
        n = aw + bw
        return None if n == 0 else aw / n

    names = [name for name, _ in shapes.most_common()]
    drag = "Dragapult"
    drag_row = {}
    for b in names:
        if b == drag:
            continue
        r = rate(drag, b)
        if r is not None:
            drag_row[b] = {"win_rate": round(r, 3), "games": sum(rows[drag][b])}
    # Who beats Dragapult, from the other side's perspective.
    beats = {}
    for a in names:
        if a == drag:
            continue
        r = rate(a, drag)
        if r is not None:
            beats[a] = {"win_rate": round(r, 3), "games": sum(rows[a][drag])}

    payload = {
        "episodes": episodes,
        "archetype_decks": dict(shapes.most_common()),
        "dragapult_share": round(shapes[drag] / max(1, sum(shapes.values())), 4),
        "dragapult_as_a": drag_row,
        "who_beats_dragapult": dict(sorted(beats.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)),
        "matrix": {a: {b: rows[a][b] for b in names if sum(rows[a][b])}
                   for a in names if any(sum(rows[a][b]) for b in names)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"episodes={episodes} archetypes={len(shapes)} dragapult_share={payload['dragapult_share']}")
    print("who beats Dragapult:")
    for k, v in list(payload["who_beats_dragapult"].items())[:12]:
        print(f"  {k:32s} {v['win_rate']:.1%} ({v['games']} games)")
    print("Dragapult vs:")
    for k, v in sorted(drag_row.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)[:12]:
        print(f"  {k:32s} {v['win_rate']:.1%} ({v['games']} games)")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
