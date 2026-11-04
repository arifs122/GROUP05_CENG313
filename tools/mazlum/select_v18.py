"""Select the v18 "smart" Kaggle-meta subset: only episodes Banette WON.

Sources:
  * every episode of our latest submission (``--mined-ids``), and
  * episodes mined from top-N leaderboard teams (default 200),

and in both cases only episodes whose winner side plays Banette. This is the
anti-echo-chamber filter: successful strategies only, from strong pilots.

Usage::

    python tools/mazlum/select_v18.py --mined-ids data/mazlum/v18_mined_ids.json \
        --out data/mazlum/v18_selected_ids.json --top-teams 200
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tools", ROOT / "tools" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_replays as cr  # noqa: E402

MANIFEST = ROOT / "replays" / "_collected.json"
TARGET = "Banette"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select Banette-win episodes for v18")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--mined-ids", type=Path, default=ROOT / "data" / "mazlum" / "v18_mined_ids.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mazlum" / "v18_selected_ids.json")
    parser.add_argument("--top-teams", type=int, default=200)
    parser.add_argument("--include-others", action="store_true",
                        help="include Banette wins from every team, not just the top N")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes") or []
    mined = {int(x) for x in json.loads(args.mined_ids.read_text(encoding="utf-8"))}

    top_teams: set[str] = set()
    if not args.include_others and args.top_teams > 0:
        for _team_id, name in cr.list_teams(args.top_teams):
            top_teams.add(name)
    print(f"mined ids: {len(mined)}  top teams: {len(top_teams) or 'ALL'}", flush=True)

    selected: list[int] = []
    from_ours = from_top = 0
    opponents = collections.Counter()
    skipped_no_winner = skipped_loss = 0

    for entry in episodes:
        arch = entry.get("archetypes") or []
        winner = entry.get("winner_side")
        if TARGET not in arch:
            continue
        if winner is None or len(arch) < 2:
            skipped_no_winner += 1
            continue
        if arch[winner] != TARGET:
            skipped_loss += 1
            continue

        eid = int(entry["episode_id"])
        ours = eid in mined
        strong = entry.get("team") in top_teams
        if not (ours or strong):
            continue

        selected.append(eid)
        from_ours += int(ours)
        from_top += int(strong)
        loser = arch[1 - winner]
        opponents[loser] += 1

    selected.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected), encoding="utf-8")

    print(f"\nselected episodes: {len(selected)}  (ours {from_ours} / top-team {from_top})")
    print(f"skipped: no winner {skipped_no_winner}  Banette losses {skipped_loss}")
    print("beat:", opponents.most_common(10))
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
