#!/usr/bin/env python3
"""Deep, turn-level behavior analysis for Petit Canard's Dragapult."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as rm  # noqa: E402

TEAM = "Petit Canard"
FOLDER = ROOT / "replays" / "top2_policy"


def main():
    games = []
    for path in sorted(FOLDER.glob("episode-*.json")):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = rm.team_names(e)[:2]
        ours = [i for i, n in enumerate(names) if TEAM in str(n)]
        if len(ours) != 1:
            continue
        me = ours[0]
        winner = rm.winner_index(e)
        decks = rm.extract_decks(e)
        if not decks:
            continue
        result = "W" if winner == me else "L"
        turns = defaultdict(list)
        for player, step, obs, sel, action in rm.iter_decisions(e):
            if player != me:
                continue
            turn = int((obs.get("current") or {}).get("turn", 0) or 0)
            options = sel.get("option") or []
            for idx in action or []:
                if not isinstance(idx, int) or idx < 0 or idx >= len(options):
                    continue
                label = rm.option_label(obs, player, options[idx])
                kind = label.split(" ", 1)[0]
                if kind in {"PLAY", "ATTACH", "EVOLVE", "ABILITY", "ATTACK", "RETREAT"}:
                    turns[turn].append(label)
        state = None
        for step in reversed(e.get("steps", [])):
            if me >= len(step):
                continue
            cur = (step[me].get("observation") or {}).get("current")
            if cur:
                m = cur["players"][me]
                state = {"turn": cur.get("turn"), "deck": m.get("deckCount"),
                         "hand": len(m.get("hand") or []), "prizes": len(m.get("prize") or []),
                         "bench": len(m.get("bench") or [])}
                break
        games.append({"id": e["info"]["EpisodeId"], "res": result, "opp": rm.archetype(decks[1 - me]),
                      "turns": dict(turns), "state": state or {}})

    print("games", len(games), "wins", sum(g["res"] == "W" for g in games))
    phases = {"early": (1, 4), "mid": (5, 9), "late": (10, 99)}
    for label in ("W", "L"):
        subset = [g for g in games if g["res"] == label]
        print(f"\n=== {label} n={len(subset)} ===")
        for phase, (lo, hi) in phases.items():
            counts = Counter()
            for g in subset:
                for turn, actions in g["turns"].items():
                    if lo <= turn <= hi:
                        counts.update(actions)
            games_n = max(1, len(subset))
            print(phase, {k: round(v / games_n, 2) for k, v in counts.most_common(15)})
        # per-game ability / key action averages
        key = Counter()
        for g in subset:
            all_actions = [a for acts in g["turns"].values() for a in acts]
            for a in all_actions:
                if a.startswith("ABILITY") or a.startswith("PLAY Boss") or a.startswith("PLAY Crispin") \
                        or a.startswith("PLAY Judge") or a.startswith("PLAY Risky") or a.startswith("PLAY Team Rocket") \
                        or a.startswith("ATTACK"):
                    key[a] += 1
        print("key/game", {k: round(v / games_n, 2) for k, v in key.most_common(20)})

    # Games using Munkidori ability, by result and count.
    for threshold in (0, 1, 3, 5):
        sub = []
        for g in games:
            n = sum(1 for acts in g["turns"].values() for a in acts if a.startswith("ABILITY Munkidori"))
            if n >= threshold:
                sub.append(g)
        w = sum(g["res"] == "W" for g in sub)
        print(f"Munkidori>={threshold}: n={len(sub)} wr={w/max(1,len(sub)):.1%}")

    # Sample long wins/losses, mirror/Kangaskhan preferred.
    for label in ("W", "L"):
        candidates = [g for g in games if g["res"] == label and (g["state"].get("turn") or 0) >= 10
                      and g["opp"] in {"Dragapult", "Mega Kangaskhan"}]
        for g in candidates[:3]:
            print(f"\n--- {label} {g['id']} vs {g['opp']} final={g['state']} ---")
            for turn in sorted(g["turns"]):
                print(f"t{turn:02d}: " + " | ".join(g["turns"][turn]))

    out = ROOT / "results" / "top2_deep_analysis.json"
    out.write_text(json.dumps(games, indent=2), encoding="utf-8")
    print("\nwritten", out)


if __name__ == "__main__":
    main()
