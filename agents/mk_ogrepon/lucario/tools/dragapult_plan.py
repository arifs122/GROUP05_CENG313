#!/usr/bin/env python3
"""Extract early-game patterns from Mega Lucario vs Dragapult ladder replays."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as rm  # noqa: E402


def name(cid):
    return rm.card_name(cid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, default=ROOT / "replays")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "agents" / "mk_ogrepon" / "lucario" / "reports" / "dragapult_plan.json")
    args = parser.parse_args()

    stats = {"win": defaultdict(Counter), "loss": defaultdict(Counter)}
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
        luc = [i for i, d in enumerate(decks) if 678 in d]
        dra = [i for i, d in enumerate(decks) if 121 in d or 120 in d]
        if len(luc) != 1 or len(dra) != 1 or luc[0] == dra[0]:
            continue
        me = luc[0]
        episodes += 1
        side = "win" if winner == me else "loss"
        st = stats[side]
        first_active = ""
        first_bench = []
        for player, si, obs, sel, act in rm.iter_decisions(e):
            if player != me or not sel or int(sel.get("type", -1)) != 0:
                continue
            cur = obs.get("current") or {}
            turn = int(cur.get("turn", 0) or 0)
            if turn > 4:
                continue
            pl = (cur.get("players") or [{}, {}])[me]
            if not first_active and pl.get("active"):
                a = pl["active"][0]
                if a:
                    first_active = name(a["id"])
            if not first_bench:
                first_bench = [name(p["id"]) for p in pl.get("bench", []) if p]
            for idxx in act:
                opts = sel.get("option") or []
                if not (0 <= idxx < len(opts)):
                    continue
                lab = rm.option_label(obs, me, opts[idxx])
                kind = lab.split(" ", 1)[0]
                st["actions"][kind] += 1
                if lab.startswith("PLAY "):
                    st["plays"][lab[5:]] += 1
                elif lab.startswith("ATTACK "):
                    st["attacks"][lab[7:]] += 1
                elif lab.startswith("ABILITY "):
                    st["abilities"][lab[8:]] += 1
                elif lab.startswith("ATTACH "):
                    target = lab.split(" -> ", 1)[-1]
                    st["attach_targets"][target] += 1
                if turn <= 3:
                    st["t1_3_actions"][kind] += 1
        st["first_active"][first_active] += 1
        st["first_bench"]["|".join(first_bench)] += 1

    out = {
        "episodes": episodes,
        "win": {k: dict(v.most_common()) for k, v in stats["win"].items()},
        "loss": {k: dict(v.most_common()) for k, v in stats["loss"].items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"episodes={episodes} wins={sum(stats['win']['first_active'].values())} "
          f"losses={sum(stats['loss']['first_active'].values())}")
    for side in ("win", "loss"):
        print(f"\n## {side}")
        for key in ("first_active", "first_bench", "t1_3_actions", "attach_targets",
                    "plays", "attacks", "abilities"):
            vals = stats[side][key].most_common(10)
            print(key, vals)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
